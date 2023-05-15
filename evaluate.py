import os
import csv
import json
import logging
import argparse
import shutil
from dotenv import load_dotenv
from src.preprocessing import read_json_file
from sklearn.metrics import classification_report

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

def parse_arguments():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_path', type=str, required=True, help='Path to the raw data folder')
    arg_parser.add_argument('--exp_path',type=str ,help='Path to the experiment directory')
    arg_parser.add_argument('--out_path', type=str, required=True, help='Pathto directory where to save scores')
    arg_parser.add_argument('-ood_val', '--ood_validation', action='store_true', default=False, help='Whether OOD validation was used')
    arg_parser.add_argument('-d', '--domains', type=str, nargs='+', default=['ai', 'literature', 'music', 'news', 'politics', 'science'], help="list of domains")
    arg_parser.add_argument('-rs', '--seeds', type=int, nargs='+', help='Seeds used / to average over')
    arg_parser.add_argument('-map', '--mapping_types', type=str, nargs='+', help='Mappings used in training evaluation will go through all, one of None, "manual", "elisa", "embedding", "ood_clustering", "topological", "thesaurus_affinity". ood_clustering can only be used if --ood_validation is True.')

    return arg_parser.parse_args()

def get_metrics(gold_path, predicted_path):

    # setup label types
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS").split())}

    _, _, _, gold_relations = read_json_file(gold_path, label_types, multi_label=True)

    # get the predicted labels
    predicted = []
    with open(predicted_path) as predicted_file:
        predicted_reader = csv.reader(predicted_file, delimiter=',')
        next(predicted_reader)
        for line in predicted_reader:
            pred_instance = [0] * len(label_types.keys())
            for rel in line[0].split(' '):
                pred_instance[label_types[rel]] = 1
            predicted.append(pred_instance)
    
    assert len(gold_relations) == len(predicted), "Length of gold and predicted labels should be equal."

    labels = os.getenv(f"RELATION_LABELS").split()
    report = classification_report(gold_relations, predicted, target_names=labels, output_dict=True, zero_division=0)

    # do not consider the labels with 0 instances in the test set in the macro-f1 computation
    macro = sum([elem[1]['f1-score'] if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()]) / sum([1 if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()])

    return report, macro


if __name__ == '__main__':

    args = parse_arguments()
    # Sort domain list so it's always in the same order (important when saving predictions -> order of sentences)
    args.domains.sort()

    domain_list = "".join([domain[0] for domain in sorted(args.domains)])
    exp_type = "ood_validation" if args.ood_validation else "all"

    for mapping_type in args.mapping_types:
        if mapping_type is None or mapping_type == "None":
            mapping_type = "no_mapping"
        if exp_type == "all" and mapping_type == "ood_clustering":
            logging.info("OOD clustering mapping type not possible without OOD eval, skipping.")
            continue
        # Aggregator for mapping type
        results = {}
        for seed in args.seeds:
            # Folder where predictions went
            exp_path = os.path.join(args.exp_path, f"{domain_list}_{seed}", f"{mapping_type}", exp_type)

            if exp_type == "ood_validation":
                # For each random seed let's make a dictionary for the results
                results[seed] = {}
                # Besides domains, let's calculate average of all domains for one given random seed
                results[seed]["average"] = {}
                # For each test domain
                for domain in args.domains:
                    pred_path = os.path.join(exp_path, domain, "predictions.csv")
                    # Gold label path
                    if mapping_type == "ood_clustering":
                        gold_path = os.path.join(args.exp_path, "ood_clustering_data", f"{seed}", f"{domain}-test.json")
                    else:
                        gold_path = os.path.join(args.data_path, f"{domain}-test.json")
                    
                    logging.info(f"Evaluating {gold_path} and {pred_path}.")
                    domain_metrics, macro = get_metrics(gold_path, pred_path)
                    # Replace macro f1 measure with the one that removes 0 support labels (as in CrossRE)
                    # Set precision and recall to invalid value (since those are not replaced)
                    domain_metrics["macro avg"]["precision"] = 999999
                    domain_metrics["macro avg"]["recall"] = 999999
                    domain_metrics["macro avg"]["f1-score"] = macro
                    logging.info(f"Saving scores to {os.path.join(exp_path, domain, 'metrics.json')}")
                    # Save metrics for given domain in its folder
                    with open(os.path.join(exp_path, domain, "metrics.json"), "w") as f:
                        json.dump(domain_metrics, f, indent=4)
                    # Add it to the metrics dict
                    results[seed][domain] = domain_metrics

                    # Let's gradually build the averages:
                    #   1. results[domain]: For each domain, aggregate results over random seeds
                    #   2. results[seed]["average"]: Average of all domains for one given random seed

                    if domain not in results:
                        results[domain] = {}
                    # Metric group: like 'macro avg' or relationship label like 'cause-effect'
                    for metric_group, metrics in domain_metrics.items():
                        # For type 1 and type 2 aggregates, create the metric group dict if doesn't yet exists
                        if metric_group not in results[seed]["average"]:
                            results[seed]["average"][metric_group] = {}
                        if metric_group not in results[domain]:
                            results[domain][metric_group] = {}
                        
                        # Metric: Precision, Recall, F1, Support
                        for metric, value in metrics.items():
                            # If metric not in aggregates, add it
                            if metric not in  results[seed]["average"][metric_group]:
                                results[seed]["average"][metric_group][metric] = []
                            if metric not in results[domain][metric_group]:
                                results[domain][metric_group][metric] = []
                            # We use lists, because we only add the value, if the label actually occured (support != 0)
                            if metrics["support"] != 0:
                                results[seed]["average"][metric_group][metric].append(value)
                                results[domain][metric_group][metric].append(value)

                # Go through the type 2 aggregate and average it out
                for metric_group, metrics in results[seed]["average"].items():
                    for metric, value in metrics.items():
                        # If there actually were examples with that label
                        if len(value) != 0:
                            # Support: we do the sum
                            if metric == "support":
                                metrics[metric] = sum(value)
                            # Otherwise avg
                            else:
                                metrics[metric] = sum(value) / len(value)
                        # If no examples -> 0 value
                        else:
                            metrics[metric] = 0
                # Save type 2 aggregate in it's folder
                with open(os.path.join(exp_path, "metrics.json"), "w") as f:
                    json.dump(results[seed]["average"], f, indent=4)
            
            # If 'all' (not ood eval)
            else:
                temp_path = os.path.join(exp_path, "temp.json")
                shutil.copyfile(os.path.join(args.data_path, f"{args.domains[0]}-test.json"), temp_path)
                with open(temp_path, "a") as f:
                    for domain in args.domains[1:]:
                        with open(os.path.join(args.data_path, f"{domain}-test.json")) as f2:
                            for line in f2.readlines():
                                f.write(line)
                pred_path = os.path.join(exp_path, "predictions.csv")
                gold_path = temp_path
                logging.info(f"Evaluating {gold_path} and {pred_path}.")
                metrics, macro = get_metrics(gold_path, pred_path)
                metrics["macro avg"]["precision"] = 999999
                metrics["macro avg"]["recall"] = 999999
                metrics["macro avg"]["f1-score"] = macro
                logging.info(f"Saving scores to {os.path.join(exp_path, 'metrics.json')}")
                with open(os.path.join(exp_path, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                results[seed] = metrics
                os.remove(temp_path)


        # AFTER WE WENT THROUGH ALL THE SEEDS
        # This will be the cross-seed summary (we save seed list otherwise we wouldn't know across which seeds the values were calculated)
        to_write = {"seeds": args.seeds}
        # If OOD validation: go through each domain and aggregate across seeds (type 1 aggregate)
        if exp_type == "ood_validation":
            for domain in args.domains:
                for metric_group, metrics in results[domain].items():
                    for metric, value in metrics.items():
                        # If there actually were examples with that label
                        if len(value) != 0:
                            metrics[metric] = sum(value) / len(value)
                        # If no examples -> 0 value
                        else:
                            metrics[metric] = 0

            # Let's calculate overall avg (through seeds and domains)
            # So this is aggregating through the type 1 aggregate (average across-seed domain results)
            # Same technique as before: Add results to lists
            results["overall_avg_through_type1"] = {}
            for domain in args.domains:
                for metric_group, metrics in results[domain].items():
                    if metric_group not in results["overall_avg_through_type1"]:
                        results["overall_avg_through_type1"][metric_group] = {}
                    for metric, value in metrics.items():
                        if metric not in results["overall_avg_through_type1"][metric_group]:
                            results["overall_avg_through_type1"][metric_group][metric] = []
                        if metrics["support"] != 0:
                            results["overall_avg_through_type1"][metric_group][metric].append(value)
            # Then average the list
            for metric_group, metrics in results["overall_avg_through_type1"].items():
                for metric, value in metrics.items():
                    # If there actually were examples with that label
                    if len(value) != 0:
                        # Support: we do the sum
                        if metric == "support":
                            metrics[metric] = sum(value)
                        # Otherwise avg
                        else:
                            metrics[metric] = sum(value) / len(value)
                    # If no examples -> 0 value
                    else:
                        metrics[metric] = 0

            # Let's calculate overall avg (through seeds and domains)
            # So this is aggregating through the type 2 aggregate
            # Same technique as before: Add results to lists
            results["overall_avg_through_type2"] = {}
            for seed in args.seeds:
                for metric_group, metrics in results[seed]["average"].items():
                    if metric_group not in results["overall_avg_through_type2"]:
                        results["overall_avg_through_type2"][metric_group] = {}
                    for metric, value in metrics.items():
                        if metric not in results["overall_avg_through_type2"][metric_group]:
                            results["overall_avg_through_type2"][metric_group][metric] = []
                        if metrics["support"] != 0:
                            results["overall_avg_through_type2"][metric_group][metric].append(value)
            # Then average the list
            for metric_group, metrics in results["overall_avg_through_type2"].items():
                for metric, value in metrics.items():
                    # If there actually were examples with that label
                    if len(value) != 0:
                        # Support: we do the sum
                        if metric == "support":
                            metrics[metric] = sum(value)
                        # Otherwise avg
                        else:
                            metrics[metric] = sum(value) / len(value)
                    # If no examples -> 0 value
                    else:
                        metrics[metric] = 0
            
            # AAAAAND IT SEEMS THE TYPE1 AND TYPE2 AGGREGATES AGREE!

            to_write["average"] = results["overall_avg_through_type2"]
            for domain in args.domains:
                to_write[domain] = results[domain]

        # If not OOD eval
        else:
            # Let's calculate overall avg (through seeds and domains)
            # Same technique as before: Add results to lists
            results["overall_avg_through_type2"] = {}
            for seed in args.seeds:
                for metric_group, metrics in results[seed].items():
                    if metric_group not in results["overall_avg_through_type2"]:
                        results["overall_avg_through_type2"][metric_group] = {}
                    for metric, value in metrics.items():
                        if metric not in results["overall_avg_through_type2"][metric_group]:
                            results["overall_avg_through_type2"][metric_group][metric] = []
                        if metrics["support"] != 0:
                            results["overall_avg_through_type2"][metric_group][metric].append(value)
            # Then average the list
            for metric_group, metrics in results["overall_avg_through_type2"].items():
                for metric, value in metrics.items():
                    # If there actually were examples with that label
                    if len(value) != 0:
                        # Support: we do the sum
                        if metric == "support":
                            metrics[metric] = sum(value)
                        # Otherwise avg
                        else:
                            metrics[metric] = sum(value) / len(value)
                    # If no examples -> 0 value
                    else:
                        metrics[metric] = 0

            to_write["average"] = results["overall_avg_through_type2"]

        save_dir = os.path.join(args.out_path, f"{domain_list}_{mapping_type}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{exp_type}.json"), "w") as f:
            json.dump(to_write, f, indent=4)

import os
import sys
import csv
import json
import logging
import argparse
from dotenv import load_dotenv
from src.preprocessing import read_json_file
from sklearn.metrics import classification_report

# python evaluate.py --gold_path data/crossre_data/ai-test.json --pred_path data/predictions/almnps_4012/elisa/ood_validation/ai/predictions.csv --out_path test/ --summary_exps test/summary.json

# TODO args.domains.sort()

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

def parse_arguments():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--gold_path', type=str, nargs='?', required=True, help='Path to the gold labels file')
    arg_parser.add_argument('--pred_path', type=str, nargs='?', required=True, help='Path to the predicted labels file')
    arg_parser.add_argument('--out_path', type=str, nargs='?', required=True, help='Path where to save scores')
    arg_parser.add_argument('--summary_exps', type=str, nargs='?', required=True, help='Path to the summary of the overall experiments')
    
    arg_parser.add_argument('--mapping_method', type=str, nargs='?', default="no_mapping", help='Mapping method: use name of folder')
    arg_parser.add_argument('--ood', type=bool, nargs='?', default=False, help='Wether it is out of domain evaluation or not')
    arg_parser.add_argument('--domains', type=list, nargs='?', default=['ai', 'literature', 'music', 'news', 'politics', 'science'], help='Name of the experiment')
    
    return arg_parser.parse_args()

def get_metrics(gold_path, predicted_path, ood=True, domains=['ai', 'literature', 'music', 'news', 'politics', 'science']):

    # setup label types
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS").split())}

    if ood:
        _, _, _, gold_relations = read_json_file(gold_path, label_types, multi_label=True)
    else:
        gold_relations = []
        for d in domains:
            g_path = os.path.join(gold_path,f"{d}-test.json")
            _, _, _, g = read_json_file(g_path, label_types, multi_label=True)
            gold_relations += g

    # get the predicted labels
    predicted = []
    with open(predicted_path, encoding="UTF-8") as predicted_file:
        predicted_reader = csv.reader(predicted_file, delimiter=',')
        next(predicted_reader)
        for line in predicted_reader:
            pred_instance = [0] * len(label_types.keys())
            for rel in line[0].split(' '):
                pred_instance[label_types[rel]] = 1
            predicted.append(pred_instance)
    
        assert len(gold_relations) == len(predicted), f"Length of gold: {len(gold_relations)} != Length of pred: {len(predicted) }\n at pred path: {predicted_path}"

    labels = os.getenv(f"RELATION_LABELS").split()
    report = classification_report(gold_relations, predicted, target_names=labels, output_dict=True, zero_division=0)

    # do not consider the labels with 0 instances in the test set in the macro-f1 computation
    macro = sum([elem[1]['f1-score'] if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()]) / sum([1 if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()])

    return report, macro


if __name__ == '__main__':

    args = parse_arguments()
    # Sort domain list so it's always in the same order (important when saving predictions -> order of sentences)
    args.domains.sort()

    if not args.ood:

        if args.mapping_method == "ood_clustering":
            logging.info("Exiting: No evaluation for ood_clustering with all domains at once")
            sys.exit()
        else:
            logging.info("Evaluating Predictions for All Domains at Once")
            pred_path = os.path.join(args.pred_path, args.mapping_method, "all", "predictions.csv")
            saving_path = os.path.join(args.pred_path, args.mapping_method, "all", "evaluation.json")
            metrics, macro = get_metrics(args.gold_path, pred_path, ood=args.ood, domains=args.domains)

            logging.info(f"Writing metrics to {saving_path}")
            json.dump(metrics, open(saving_path, "w"))

            # write summary statistics to file
            with open(args.summary_exps, 'a') as file:
                file.write(f"Metrics for all domains with mapping method {args.mapping_method}\n")
                file.write(f"Micro F1: {metrics['micro avg']['f1-score'] * 100}\n")
                file.write(f"Macro F1: {macro * 100}\n")
                file.write(f"Weighted F1: {metrics['weighted avg']['f1-score'] * 100}\n")
                file.write("--------------------------------------------------------------\n")        
    else:
        if args.mapping_method == "elisa":
            logging.info("Exiting: No evaluation for elisa with ood")
            sys.exit()
        logging.info("Evaluating Predictions for OOD")
        for domain in args.domains:

            logging.info(f"Evaluating domain {domain}")

            gold_path_domain = os.path.join(args.gold_path,f"{domain}-test.json") # args.gold_path is a folder
            
            ## there is no domains in the all evaluation so only oodValidation
            pred_path_domain = os.path.join(args.pred_path, args.mapping_method, "ood_validation", domain, "predictions.csv")
            saving_path = os.path.join(args.pred_path, args.mapping_method, "ood_validation", domain, "evaluation.json")

            # get metrics
            if os.path.isfile(pred_path_domain) and (args.mapping_method != "elisa" and args.ood) :
            
                metrics, macro = get_metrics(gold_path_domain, pred_path_domain)
            

                # write metrics to file
                json.dump(metrics, open(saving_path, "w"))

                # write summary statistics to file
                with open(args.summary_exps, 'a') as file:
                    file.write(f"Metrics for {domain}-domain with mapping method {args.mapping_method}\n")
                    file.write(f"Micro F1: {metrics['micro avg']['f1-score'] * 100}\n")
                    file.write(f"Macro F1: {macro * 100}\n")
                    file.write(f"Weighted F1: {metrics['weighted avg']['f1-score'] * 100}\n")
                    file.write("--------------------------------------------------------------\n")
                logging.info(f"Evaluation complete wrote metrics to {saving_path}")
            else:
                
                logging.info(f"No metrics found in: {pred_path_domain}")
                
                with open(args.summary_exps, 'a') as file:
                    file.write("<-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!>\n")
                    file.write(f"No metrics for {domain}-domain with mapping method {args.mapping_method}\n")
                    file.write(f"No predictions in this path: {pred_path_domain} \n")
                    file.write("--------------------------------------------------------------\n")

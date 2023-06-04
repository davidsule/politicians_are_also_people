# Politicians Are Also People: Mapping Is All You Need
###### Clustering Entity Types in Cross-Domain Relation Classification Setups
This repistory contains the data, code, and paper for the Second Year Project assignment of David Peter Süle, Mie Jonasson, and Nicklas Koch Rasmussen at the IT University of Copenhagen.

## Research Question
*What is the performance impact of clustering domain-specific named entity types in cross-domain relation-classification setups and what benchmark can be established for future research?*

## Abstract
Relation Extraction is an evolving field within natural language processing. As its last step, Relation Classification (RC) aims to identify the relation type to which two semantically related named entities belong. Cross-domain setups are especially challenging, even more so when domain-specific entity types are used. Research is scarce in the area and mostly focuses on using generic entity types or simply fine-tune the model on a single target domain. This might still offer challenges, 
when annotated data is not accessible for fine-tuning.

In this paper we explore ways of clustering domain-specific named entity types to reduce cross-domain complexity and improve performance on previously unseen domains. We propose five different methods of grouping entity types and evaluate them in multi-domain and out-of-domain scenarios using our two new benchmarks. In conclusion, we find that all our entity mapping methods outperform the baseline in the out-of-domain setting, with the best performing model improving on the baseline by $8.6$ percentage points in weighted F1.

## Attribution
Our work relied heavily on the CrossRE project by Elisa Bassignana and Barbara Plank: [CrossRE: A Cross-Domain Dataset for Relation Extraction](https://aclanthology.org/2022.findings-emnlp.263) (Bassignana & Plank, Findings 2022), and [their repistory](https://github.com/mainlp/CrossRE).


## How to run the project

### Installing requirements
```bash
pip install -r requirements.txt

```

### Run training, predictions, and calculate results
```bash
./run.sh
```

## Folder Structure
- data
    - crossre_data
        - The training-, development- and test-data as provided by the [CrossRE](https://github.com/mainlp/CrossRE) project.
    - predictions
        - names of folders: DOMAIN-LIST_SEED where the domain list is abbreviated from the first letter of the domains used during training; *contains predictions produced by running main script*.
        - ood_clustering_data: data for training with OOD clustering method.
    - results: Aggregated results
- figures: images / plots used for the report.
- src
    - Scripts used for training. These are mainly supplied by the CrossRE project, with slight modifications.
- util
    - Helper functions to check validity of results.

(Note: 'ood validation' stands for OOD evaluation and 'all' stands for the multi-domain results in the file names.)

## Cite
```
@misc{politicians-are-people,
  title        = "Politicians Are Also People: Mapping Is All You Need",
  author       = "S{\"u}le, David Peter and Jonasson, Mie and Rasmussen, Nicklas Koch",
  howpublished = "\url{https://github.itu.dk/dasy/2yp_project}",
  year         = "2023",
  school       = "IT University of Copenhagen",
  address      = "Copenhagen, Denmark",
  note         = "Second Year Project course report",
  abstract     = "Relation Extraction is an evolving field within natural language processing. As its last step, Relation Classification (RC) aims to identify the relation type to which two semantically related named entities belong. Cross-domain setups are especially challenging, even more so when domain-specific entity types are used. Research is scarce in the area and mostly focuses on using generic entity types or simply fine-tune the model on a single target domain. This might still offer challenges, when annotated data is not accessible for fine-tuning. In this paper we explore ways of clustering domain-specific named entity types to reduce cross-domain complexity and improve performance on previously unseen domains. We propose five different methods of grouping entity types and evaluate them in multi-domain and out-of-domain scenarios using our two new benchmarks. In conclusion, we find that all our entity mapping methods outperform the baseline in the out-of-domain setting, with the best performing model improving on the baseline by 8.6 percentage points in weighted F1.
}
```

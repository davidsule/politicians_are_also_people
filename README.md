# Second Year Project
## Research Question 
*Can we group entities in a meaningful way in cross-domain setups and inject these groupings to improve performance?*

## How to run the project

### Installing requirements
```bash
pip install -r requirements.txt

```

### Run training, predictions and calculate f1-scores
```bash
bash run.sh
```

## File Structure
- data
    - crossre_data
        - The training-, development- and test-data as provided by the CrossRE project: [CrossRE](https://github.com/mainlp/CrossRE).
    - predictions
        - names of folders: DOMAIN-LIST_SEED where the domain list is abbreviated from the first letter in the domain; *contains predictions produced by running main script*.
        - ood_clustering_data: test data with sentences removed (only used for OOD-clustering).
    - results 
        - The overall F1-scores for each mapping method.
- figures
    - images / plots used for the report. (can be reproduced using 'plotting_results.ipynb')
- src
    - Scripts used for training. These are mainly supplied by the [CrossRE](https://github.com/mainlp/CrossRE) project, with modifications as described in report.
- util
    - Helper functions to check validity of results.

<!-- 
## To-dos
- Add requirements.txt
- Discuss eval of OOD clustering: diff nr of unique entitites + no unique to news
- Rewrite README for reader of the project -->
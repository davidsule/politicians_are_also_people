# 2yp_project
Second Year Project repo

## Research Question 
Can we group entities in a meaningful way in cross-domain setups and inject these groupings to improve performance?

## File Structure
- data
    - crossre_data
        - The training-, developing- and test-data
    - predictions
        - names of folders: DOMAIN-LIST_SEED where the domain list is abbreviated from the first letter in the domain 
    - results 
        - The results of the predictions
- figures
    - images of plotting.
- src
    - with all the script files
- util
    - helper functions to check results


## How to run the training

Download requirements
```bash
pip install -r requirements.txt

```

Run training, predictions and calculate f1-scores
```bash
bash run.sh
```


## To-dos
- Add requirements.txt
- Add licence
- Discuss eval of OOD clustering: diff nr of unique entitites + no unique to news
- Rewrite README for reader of the project
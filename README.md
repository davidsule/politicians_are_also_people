# 2yp_project
Second Year Project repo

## File Structure
- Baseline
    - Contain a notebook explaining steps used to train a baseline classifier and produce baseline predictions on test data.
- CrossRE
    - Contains the CrossRE dataset and project scripts for a baseline classifier. Note, that we have modified some files to fit our objective and needs.
- project_rob
    - Contains project description and requirements.

## To-dos
- Add requirements.txt

## Clustering of entity types
### Sources
1. Manual Grouping
2. Word2vec
    a. Embedding-based
    b. Distance (similarity)-based
3. Wordnet
    a. Hypernym-based (select 2nd, 3rd ... level)
    b. Distance (similarity)-based
### Clustering methods
- K-moids
- DBSCAN
- Spectral
- Affinity Propagation
- Some hierarchical method

## Project Work Schedule
- Create Groups / Hierarchy (Manual task)
- Create Groups / Hierarchy (Automatically / wordnet)
- Properly define baseline (train on 5, validate on last?)
- Determine test paramaters
- Execute testing
- Create visuals / results for report
- Write Report

## Brain Storm Questions
 - which models are suited for cross-domain? 
 - What can be Improved in current methods?
 - How to make a baseline model? (Trained seperately per domain?)
 - How to handle labels in cross-domain?
 - What are we actually even trying to classify?

### RC Project Proposal 3
Can we SOMEHOW find a way to represent different entity types in different domains (i.e. use entity types from Reuters / Literature in the Music Domain) in the same subset of labels? 

How do we Build Hierarchy of labels, that can represent entities and their relations across domains? 

We want to PREDICT the label from a *specific subset* by some method, that still maintains the structure and meaning of extracted relations. That still remains well-performing, and thus is generalizable.
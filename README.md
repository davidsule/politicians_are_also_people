# 2yp_project
Second Year Project repo

## File Structure
- TBA
- CrossRE
    - Contains the CrossRE dataset and project scripts for reference.
- project_rob
    - Contains project description and requirements.

## To-dos
- Add requirements.txt
- Add licence
- Discuss eval of OOD clustering: diff nr of unique entitites + no unique to news
- Rewrite README for reader of the project

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
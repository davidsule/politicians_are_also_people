#!/bin/bash

SEEDS=( 4012 5096 8878 8857 9908 )
MAPPING=( "no_mapping" "manual" "embedding" "ood_clustering" "topological" "thesaurus_affinity" ) 
OOD=(true false)

for rs in "${!SEEDS[@]}"; do
  echo "Evaluatoin on random seed ${SEEDS[$rs]}."
    for mapping in "${!MAPPING[@]}"; do
       for ood in "${!MAPPING[@]}"; do
            # check if evaluation already exists
            # if experiment is new, train classifier
            echo "Evaluating model on random seed ${SEEDS[$rs]}."
            if [ ${MAPPING[$mapping]} == "ood_clustering" ] 
            then
              python evaluate.py \
                    --gold_path data/predictions/ood_clustering_data/${SEEDS[$rs]}/ \
                    --pred_path data/predictions/almnps_${SEEDS[$rs]}/ \
                    --out_path test/ \
                    --mapping_method ${MAPPING[$mapping]} \
                    --ood ${OOD[$ood]} \
                    --summary_exps evaluation_summary.txt 
            else
              # will run all domains by default
              python evaluate.py \
                      --gold_path data/crossre_data/ \
                      --pred_path data/predictions/almnps_${SEEDS[$rs]}/ \
                      --out_path test/ \
                      --mapping_method ${MAPPING[$mapping]} \
                      --ood ${OOD[$ood]} \
                      --summary_exps evaluation_summary.txt 
            fi
        done
    done
done
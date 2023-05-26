#!/bin/bash
DATA_PATH=data/crossre_data/
EXP_PATH=data/predictions
OUT_PATH=data/results

SEEDS=( 4012 5096 8878 8857 9908 )
MAPPING=( "None" "manual" "embedding" "ood_clustering" "topological" "thesaurus_affinity" )
for rs in "${!SEEDS[@]}"; do
        for mapping in "${!MAPPING[@]}"; do

            # No all for ood_clustering method
            if [ ${MAPPING[$mapping]} != "ood_clustering" ] 
            then
                # training
                python main.py --exp_path ${EXP_PATH} --data_path ${DATA_PATH} -rs ${SEEDS[$rs]} -shuffle -map ${MAPPING[$mapping]}
                
                # prediction
                python main.py --exp_path ${EXP_PATH} --data_path ${DATA_PATH} -rs ${SEEDS[$rs]} -shuffle -po -map ${MAPPING[$mapping]}
            fi
            
            # training
            python main.py --exp_path ${EXP_PATH} --data_path ${DATA_PATH} -rs ${SEEDS[$rs]} -shuffle -ood_val -map ${MAPPING[$mapping]}
            
            # prediction
            python main.py --exp_path ${EXP_PATH} --data_path ${DATA_PATH} -rs ${SEEDS[$rs]} -shuffle -po -ood_val -map ${MAPPING[$mapping]}

    done
done

DATA_PATH=data/crossre_data

# evaluation
python evaluate.py --data_path ${DATA_PATH} --exp_path ${EXP_PATH} --out_path ${OUT_PATH} -rs ${SEEDS[@]} -map ${MAPPING[@]}
python evaluate.py --data_path ${DATA_PATH} --exp_path ${EXP_PATH} --out_path ${OUT_PATH} -rs ${SEEDS[@]} -map ${MAPPING[@]} -ood_val
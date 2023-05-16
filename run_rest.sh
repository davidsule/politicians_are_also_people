#!/bin/bash

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map thesaurus_affinity
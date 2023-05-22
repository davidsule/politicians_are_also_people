#!/bin/bash

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map ood_clustering
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map thesaurus_affinity



# Eval command
ython evaluate.py --data_path data/crossre_data --exp_path data/predictions --out_path data/results -rs 4012 5096 8857 8878 9908 -map ood_clustering None manual embedding thesaurus_affinity topological
ython evaluate.py --data_path data/crossre_data --exp_path data/predictions --out_path data/results -rs 4012 5096 8857 8878 9908 -map ood_clustering None manual embedding thesaurus_affinity topological -ood_val
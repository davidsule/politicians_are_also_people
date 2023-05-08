#!/bin/bash

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -ood_val True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle True -po -ood_val True -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -ood_val True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle True -po -ood_val True -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -ood_val True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle True -po -ood_val True -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -ood_val True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle True -po -ood_val True -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -ood_val True -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map None
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map manual
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map elisa
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map ood_embedding
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle True -po -ood_val True -map thesaurus_affinity
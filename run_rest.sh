#!/bin/bash

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 9908 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8857 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -ood_val -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -ood_val -map thesaurus_affinity
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -ood_val -map thesaurus_affinity










python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d ai


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d literature


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d music


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d news


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d politics


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity -d science




python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d ai


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d literature


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d music


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d news


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d politics


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity -d science




python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d ai


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d literature


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d music


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d news


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d politics


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity -d science




python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d ai


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d literature


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d music


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d news


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d politics


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -map thesaurus_affinity -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 8857 -shuffle -po -map thesaurus_affinity -d science




python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d ai
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d ai


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d literature
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d literature


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d music
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d music


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d news
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d news


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d politics
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d politics


python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map None -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map manual -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map embedding -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map topological -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -map thesaurus_affinity -d science
python main.py --exp_path data/predictions/single_domain --data_path data/crossre_data/ -rs 9908 -shuffle -po -map thesaurus_affinity -d science
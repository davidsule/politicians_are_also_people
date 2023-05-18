python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 4012 -shuffle -po -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -map thesaurus_affinity

python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 5096 -shuffle -po -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -map thesaurus_affinity


python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map topological
python main.py --exp_path data/predictions --data_path data/crossre_data/ -rs 8878 -shuffle -po -map thesaurus_affinity

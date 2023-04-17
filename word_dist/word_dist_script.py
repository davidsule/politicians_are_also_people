import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from gensim.models import Word2Vec, KeyedVectors
from  gensim import downloader
import math


ENTITY_LABELS = "journal album algorithm astronomer award band book chemical conference country discipline election enzyme event field genre location magazine metrics misc artist instrument Organisation person poem politics politician product java protein researcher scientist song task theory university writer"
ENTITY_LABELS_SPLIT = ENTITY_LABELS.split()

def load_twitter_embs():


    ######### download the twitter.bin file from google drive and place it in the same folder (word_dist) ###########
    ######### https://drive.google.com/drive/folders/12vBvFPjpHx3gwkaCeuBf5MjBHdAnyk4N ###########
    try:
        twitEmbs = KeyedVectors.load_word2vec_format('twitter.bin', binary=True)
        print('loading finished')
        return twitEmbs
    except:
        print('!!! LOADING FAILED !!!')
        print('Download twitter.bin from google drive')

def load_word2vec_embs(word2vec_model_name='word2vec-google-news-300'):
    
    print('Loading word2vec model...')
    print('This can take several minutes...')
    ## This will take a very long time to download
    
    word2vec_model = downloader.load(word2vec_model_name)
    
    return word2vec_model

def distance_to_every_word(word2vec_model, word):
    
    distance_list = []
    
    for i in range(len(ENTITY_LABELS_SPLIT)):
        distance = word2vec_model.distance(word, ENTITY_LABELS_SPLIT[i])
        distance_list.append(distance)
    
    return distance_list

def get_every_embeddings(word2vec_model):
        
        embeddings = []
        for current_word in ENTITY_LABELS_SPLIT:
            embedding = word2vec_model[current_word]
            embeddings.append(embedding)
        return embeddings

def get_distance(word2vec_model, word, group):
        
        distance_list = []
        
        for i in range(len(group)):
            distance = word2vec_model.distance(word, group[i])
            distance_list.append(distance)

        return distance_list
def get_every_distance(word2vec_model):
    
    every_distance = []
    for current_word in ENTITY_LABELS_SPLIT:
        distance_list = distance_to_every_word(word2vec_model, current_word)
        every_distance.append(distance_list)
    return every_distance    

def sort_zip_labels(predictions):
    zipped = zip(ENTITY_LABELS_SPLIT, predictions)
    sorted_zipped = sorted(zipped, key=lambda x: x[1])
    return sorted_zipped

def group_by_cluster(sorted_zipped, n_clusters=5):

    ## make intial list of lists
    groups = [[] for i in range(n_clusters)]
    for word, group in sorted_zipped:
        groups[group].append(word)
    return groups

def find_label_for_cluster(word2vec_model, group):
    
    ## get distance to every word
    distances = []
    for word in group:
        distance_list = get_distance(word2vec_model, word, group)
        distances.append(distance_list)
    
    centroid = np.mean(distances, axis=1)

    ## find the word with the smallest distance to the centroid
    min_distance = 100
    min_distance_index = 0
    for current_word_co in distances:
        current_distance = math.dist(current_word_co, centroid)
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_index = distances.index(current_word_co)
    
    return group[min_distance_index]

def find_label_for_cluster_emb(word2vec_model, group):
    
    ## get distance to every word
    distance_list = word2vec_model[group]

    centroid = np.mean(distance_list, axis=0)

    ## find the word with the smallest distance to the centroid
    min_distance = 100
    min_distance_index = 0

    for i, current_word_co in enumerate(distance_list):
        current_distance = math.dist(current_word_co, centroid)
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_index = i
    
    return group[min_distance_index]

def find_vector_for_cluster(word2vec_model, group):
    ## get distance to every word
    distance_list = word2vec_model[group]

    centroid = np.mean(distance_list, axis=0)

    return centroid
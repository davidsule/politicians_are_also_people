from typing import Union
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


def find_missing(model, words):
    """Find words that are not in model.
    
    Paramteres:
        model (gensim.models.keyedvectors.KeyedVectors):  Model to
            search for the words in.
        words (list):  List with the words to check.

    Retruns:
        missing (list):  List of words not in the model.
    """
    missing = []
    for word in words:
        try:
            model[word]
        except:
            missing.append(word)
    return missing

def get_distances(model, words):
    """Get pairwise distances between words.
    
    Parameters:
        model (gensim.models.keyedvectors.KeyedVectors):  Model based on
            which distances are calculated.
        words (list):  Words to calculate distances for.

    Returns:
        distances (numpy.ndarray):  NumPy array of shape (len(words),
            len(words)), the pairwise distances.
    """
    distances = np.zeros((len(words), len(words)))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if j >= i:
                continue
            dist = model.distance(w1, w2)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def get_categories(labels, entities):
    """Get dictionary of labels as keys and the list of entities
        belonging to them as values.  The two lists must be in
        corresponding order.
    
    Parameters:
        labels (list):  List of labels.
        entities (list):  List of entity names.
    
    Returns:
        categories (dict):  Dictionary of labels as keys and the list of
            entities belonging to them as values.
    """
    categories = {}
    for label, entity in zip(labels, entities):
        if label not in categories:
            categories[label] = [entity]
        else:
            categories[label].append(entity)
    return categories

def most_frequent_synset(entities: list) -> list:
    """Return list of most frequent synset for each word in entities."""
    return list(map(lambda w: wordnet.synsets(w)[0], entities))

def minpath(synset) -> list:
    """Return the path with the minimum length to the root."""
    hypernym_paths = synset.hypernym_paths()
    minlen = len(hypernym_paths[0])
    minlen_idx = 0
    for idx, path in enumerate(hypernym_paths):
        if len(path) < minlen:
            minlen = len(path)
            minlen_idx = idx
    return hypernym_paths[minlen_idx]

def category_dict(synset_list: list, level: int = 2, words: Union[list, None] = None) -> dict:
    """Create dictionary with desired level of hypernyms as keys and
    synsets as values.  Optionally pass the corresponding list of words
    to return the words as values.

    The function searches for all the possible paths to the root and
    chooses the shortes among those for each synset in synset_list.  The
    level means the distance from the root in the path of hypernyms;
    level 0 is the root.  If the path for a synset is shorter than the
    selected level, the last level is returned.

    Each hypernym category is a key in the returned dictionary, whose
    keys are the synsets belonging to it from synset_list (or the
    corresponding words if words in not None).
    """
    categories = {}
    for i, syn in enumerate(synset_list):
        path = minpath(syn)
        try:
            cat = path[level]
        except:
            cat = path[-1]
        if cat in categories:
            if words is not None:
                categories[cat].append(words[i])
            else:
                categories[cat].append(syn)
        else:
            if words is not None:
                categories[cat] = [words[i]]
            else:
                categories[cat] = [syn]
    return categories

def get_wup_sim(entity_syns):
    """Get pairwise Wu-Palmer Similarities of the synsets from the list
    in the argument.  Returns a NumPy Array of shape (len(entity_syns),
    len(entity_syns)).
    """
    similarities = np.zeros((len(entity_syns), len(entity_syns)))
    for i, syn1 in enumerate(entity_syns):
        for j, syn2 in enumerate(entity_syns):
            if j > i:
                continue
            sim = syn1.wup_similarity(syn2)
            similarities[i, j] = sim
            similarities[j, i] = sim
    return similarities
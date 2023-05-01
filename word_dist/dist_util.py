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

def categories_from_pred(labels, entities):
    """Get dictionary of labels as keys and the list of entities
        belonging to them as values from predictions (list of labels).
        The two lists must be in corresponding order.
    
    Parameters:
        labels (list):  List of labels (predictions).
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

def name_categories(categories, model, kth_closest=6):
    """Get category dictionary out with the keys replaced with a word
    close to the mean of the embeddings of the words for that label in
    the embedding space.  The `kth_closest` word is selected to be able
    to avoid the common situation where the closest words are the same /
    too similar to one of the words - especially if there is only a
    few words in the given category.

    Parameters:
        categories (dict):  Output of `get_categories` function,
            dictionary with labels as keys and list of corresponding
            words as values.
        model (gensim.models.keyedvectors.KeyedVectors):  The model
            containing the embeddings.
        kth_closest (int):  Determines which word is selected from the
            list of 10 most similar words.  Must be in a positive int
            but less than the number of words in the model. Defaults to
            6 (the 6th most similar).
    
    Returns:
        named_categories (dict):  Category dictionary with named labels.
    """
    named_categories = {}
    for words in categories.values():
        category = model.most_similar(positive=words, topn=kth_closest)[kth_closest-1][0]
        # Handle the (extremely unlikely) case when two categories got the same name
        tmp = kth_closest + 1
        while category in named_categories:
            category = model.most_similar(positive=words, topn=tmp)[tmp-1][0]
            tmp += 1

        named_categories[category] = words
    
    return named_categories

def most_frequent_synset(entities: list) -> list:
    """Return list of most frequent synset for each word in entities."""
    return list(map(lambda w: wordnet.synsets(w)[0], entities))

def minpath(synset) -> list:
    """Return the path with the minimum length to the root.
    
    If there are several with the same minimum length the first one is
    chosen from the list returned by the `synset.hypernym_paths()`
    method.
    """
    hypernym_paths = synset.hypernym_paths()
    minlen = len(hypernym_paths[0])
    minlen_idx = 0
    for idx, path in enumerate(hypernym_paths):
        if len(path) < minlen:
            minlen = len(path)
            minlen_idx = idx
    return hypernym_paths[minlen_idx]

def graph_category_dict(synset_list: list, level: int = 2, words: Union[list, None] = None) -> dict:
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
        cat = cat.name()
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
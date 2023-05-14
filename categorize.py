import os, json
import numpy as np
from typing import Union
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from gensim import downloader
from dotenv import load_dotenv

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

from umap import UMAP

load_dotenv()


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

def name_categories(categories, model, stop_chars="._@", kth_closest=1):
    """Get category dictionary out with the keys replaced with a word
    close to the mean of the embeddings of the words for that label in
    the embedding space.  Words containig any of `stop_chars` are
    skipped and the next most similar takes their place.

    Parameters:
        categories (dict):  Output of `get_categories` function,
            dictionary with labels as keys and list of corresponding
            words as values.
        model (gensim.models.keyedvectors.KeyedVectors):  The model
            containing the embeddings.
        stop_chars (str|list|None, optional):  If closest word contains
            any of these, use next closest.  If str, Each char in it
            will be individually considered.  If list, list elements as
            a whole will be considered.  None signal no stop_chars.
            Default: '._@'.
        kth_closest (int, optional):  Determines which word is selected
            from the list of most similar ones.  Must be in a positive
            int but less than the number of words in the model.
            Defaults to 1 (the most similar one).
    
    Returns:
        named_categories (dict):  Category dictionary with named labels.
    """
    named_categories = {}
    for words in categories.values():
        category = model.most_similar(positive=words, topn=kth_closest)[kth_closest-1][0]

        # Handle the (extremely unlikely) case when two categories got the same name
        # And when the name has any of the stop_chars in it
        has_stop_char = False
        if stop_chars is not None:
            for char in stop_chars:
                if char in category:
                    has_stop_char = True
        tmp = kth_closest + 1
        while category in named_categories or has_stop_char:
            category = model.most_similar(positive=words, topn=tmp)[tmp-1][0]
            has_stop_char = False
            if stop_chars is not None:
                for char in stop_chars:
                    if char in category:
                        has_stop_char = True
            tmp += 1

        named_categories[category] = words
    return named_categories

def name_categories_old(categories, model, kth_closest=6):
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
        kth_closest (int, optional):  Determines which word is selected
            from the list of most similar ones.  Must be in a positive
            int but less than the number of words in the model.
            Defaults to 6 (the 6th most similar).
    
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

def topological_category_dict(synset_list: list, level: int = 2, words: Union[list, None] = None) -> dict:
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

    Hypernym categories have the tailing numbering removed (e.g.
    object.n.01 -> object).  If there are several different ones whose
    short form would be the same, the full name is kept for the second
    and further categories.
    """
    categories = {}
    long_forms = set()
    conflicts = set()
    # Get category name
    for i, syn in enumerate(synset_list):
        path = minpath(syn)
        try:
            cat = path[level]
        except:
            cat = path[-1]
        cat = cat.name()
        # Shorten category name (object.n.01 -> object)
        catname_to_add = cat.split(".")[0]
        # If we haven't seen the category name yet but it's short form is
        # already in the dictionary -> conflict: two different categories have
        # the same short form -> add the long name to the dict + remember the
        # conflicting category
        if cat not in long_forms and catname_to_add in categories:
            catname_to_add = cat
            conflicts.add(cat)
        # If we already know it's a conflict -> add the long name to the dict
        elif cat in conflicts:
            catname_to_add = cat
        # Remember long form
        long_forms.add(cat)

        if catname_to_add in categories:
            if words is not None:
                categories[catname_to_add].append(words[i])
            else:
                categories[catname_to_add].append(syn)
        else:
            if words is not None:
                categories[catname_to_add] = [words[i]]
            else:
                categories[catname_to_add] = [syn]
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

def entities_to_orig(categories: dict, substitutions: dict) -> dict:
    """Change substituted entity names to original ones in the final
    category dictionary.

    Parameters:
        categories (dict):  Entity cluster dictionary with category
            names as keys and list of (substituted) entities as values.
        substitutions (dict):  Dictionary with original entities as keys
            and substitutions as values.  Only needs to contain the
            modified entities.
    
    Returns:
        category_dict (dict):  The same dict as the categories input but
            with the original entity names.
    """
    # The parts with the back_substitutions is so complicated because it needs
    # to handle the situation when two of the original entities map to the same
    # new substitute entity.
    category_dict = {}
    back_substitutions = {}
    for orig_entity, substitue_entity in substitutions.items():
        if substitue_entity not in back_substitutions:
            back_substitutions[substitue_entity] = [orig_entity]
        else:
            back_substitutions[substitue_entity].append(orig_entity)
    for category, new_entity_list in categories.items():
        orig_entity_list = []
        for new_entity in new_entity_list:
            if new_entity in back_substitutions:
                for orig_entity in back_substitutions[new_entity]:
                    orig_entity_list.append(orig_entity)
            else:
                orig_entity_list.append(new_entity)
        category_dict[category] = orig_entity_list
    return category_dict

def get_manual() -> dict:
    manual_categories = {
        "person" :      ["politician", "person", "writer", "researcher", "scientist", "musicalartist"],
        "organisation": ["organisation", "politicalparty", "university", "band"],
        "location":     ["country", "location"],
        "event":        ["event", "election", "award", "conference"],
        "artifact":     ["album", "song", "academicjournal", "poem", "magazine", "book"],
        "scientific":   ["metrics", "enzyme", "protein", "chemicalcompound", "chemicalelement", "astronomicalobject"],
        "concept":      ["theory", "musicgenre", "field", "discipline", "algorithm", "literarygenre"],
        "brand":        ["product", "programlang"],
        "misc":         ["misc", "musicalinstrument", "task"]
    }
    return manual_categories

def get_elisa() -> dict:
    elisa_categories = {
        "person":       ["politician", "person", "writer", "researcher", "scientist", "musicalartist"],
        "organisation": ["organisation", "politicalparty", "university", "band"],
        "location":     ["country", "location"],
        "event":        ["event", "election", "conference"],
        "misc":         ["award", "album", "song", "academicjournal", "poem", "magazine", "book", "metrics", "enzyme", "protein", "chemicalcompound", "chemicalelement", "astronomicalobject", "theory", "musicgenre", "field", "discipline", "algorithm", "literarygenre", "product", "programlang", "misc", "musicalinstrument", "task"]
    }
    return elisa_categories

def get_embedding_based(
        entities: list,
        substitutions: Union[dict, None] = None,
        random_state: Union[int, None] = None,
        n_components: int = 35,
        n_neighbors: int = 4,
        min_dist: float = 0.3,
        damping: float = 0.5,
        name: bool = True,
        stop_chars: Union[str, list, None] = "._@",
        kth_closest: int = 1
    ) -> dict:
    """Get category dictionary based on Word2vec embeddings, with PCA
    + UMAP dimensionality reduction and AffinityPropagation clusterer.

    Parameters:
        entities (list):  Entities to cluster.
        substitutions (dict|None, optional):  What words (key) and how
            to (value) to substitute.  Default is None, i.e. no
            substitution.
        random_state (int|None, optional):  Seed for probabilistic
            components.  Default is None, no seed.
        n_components: (int, optional):  Arg for PCA, defaults to 35, but
            in every case min(n_components, len(entities)) will be used
            (if several entities are substituted with the same word,
            this number is reduced).
        n_neighbors (int, optional):  Arg for UMAP, defaults to 4.
        min_dist (float, optional):  Arg for UMAP, defaults to 0.3.
        damping (float, optional):  Arg for AffinityPropagation,
            defaults to 0.5
        name (bool, optional):  Name categories based on closest word
            in embedding space to mean of cluster?  Default: True.
        stop_chars (str|list|None, optional):  If closest word contains
            any of these, use next closest.  Only takes effect if
            name == True.  If str, Each char in it will be individually
            considered.  If list, list elements as a whole will be
            considered.  None signal no stop_chars.  Default: '._@'.
        kth_closest (int, optional):  Determines which word is selected
            from the list of most similar ones.  Must be in a positive
            int but less than the number of words in the model.
            Defaults to 1 (the most similar one). Only takes effect if
            name == True.

    Returns:
        categories (dict):  The category dictionary in category (key):
        list of entities (value) format.  The substituted entities are
        returned to the original form, even if there were several that
        mapped to the same.
    """
    entities.sort()
    w2v = downloader.load("word2vec-google-news-300")
    new_entities = []
    if substitutions is not None:
        for entity in entities:
            if entity in substitutions:
                to_add = substitutions[entity]
            else:
                to_add = entity
            if to_add not in new_entities:
                new_entities.append(to_add)
    else:
        new_entities = entities
    embeddings = w2v[new_entities]
    
    n_components = min(n_components, embeddings.shape[0])
    pipe = Pipeline([
        ("PCA", PCA(n_components=n_components, svd_solver="full")),
        ("UMAP", UMAP(n_neighbors=n_neighbors, n_components=3, metric="cosine", min_dist=min_dist, random_state=random_state)),
        ("Clusterer", AffinityPropagation(damping=damping, random_state=random_state))
    ])
    pred = pipe.fit_predict(embeddings)
    categories = categories_from_pred(pred, new_entities)
    if name:
        categories = name_categories(categories, w2v, stop_chars=stop_chars, kth_closest=kth_closest)
    categories = entities_to_orig(categories, substitutions)
    return categories

def get_ood_embedding_based(
        domain_entity_dict: dict,
        substitutions: Union[dict, None] = None,
        random_state: Union[int, None] = None,
        n_components: int = 35,
        n_neighbors: int = 4,
        min_dist: float = 0.3,
        damping: float = 0.5,
        name: bool = True,
        stop_chars: Union[str, list, None] = "._@",
        kth_closest: int = 1
    ) -> dict:
    """Get one clustering per domain that is trained leaving out the
    entities unique to that (test) domain.  The categories of left out
    entities are then predicted and they get added to the category dict.
    Category dictionary are based on Word2vec embeddings, with PCA
    + UMAP dimensionality reduction and AffinityPropagation clusterer.

    The dictionary returned has the domain names as keys, and a category
    dict for each.  These 'sub-dictionaries' have categories as keys and
    the list of entities as values.

    Parameters:
        domain_entity_dict (dict):  Dictionary with domain names as keys
            and the corresponding list of entitites as values.
        substitutions (dict|None, optional):  What words (key) and how
            to (value) to substitute.  Default is None, i.e. no
            substitution.
        random_state (int|None, optional):  Seed for probabilistic
            components.  Default is None, no seed.
        n_components: (int, optional):  Arg for PCA, defaults to 35, but
            in every case min(n_components, len(training_entities)) will
            be used, where trainig_embeddings are the set of entities
            that are in the training domains (the ones that weren't left
            out in the given ood round).  If several entities are
            substituted with the same word, this number might be
            reduced.
        n_neighbors (int, optional):  Arg for UMAP, defaults to 4.
        min_dist (float, optional):  Arg for UMAP, defaults to 0.3.
        damping (float, optional):  Arg for AffinityPropagation,
            defaults to 0.5
        name (bool, optional):  Name categories based on closest word
            in embedding space to mean of cluster?  Default: True.
        stop_chars (str|list|None, optional):  If closest word contains
            any of these, use next closest.  Only takes effect if
            name == True.  If str, Each char in it will be individually
            considered.  If list, list elements as a whole will be
            considered.  None signal no stop_chars.  Default: '._@'.
        kth_closest (int, optional):  Determines which word is selected
            from the list of most similar ones.  Must be in a positive
            int but less than the number of words in the model.
            Defaults to 1 (the most similar one). Only takes effect if
            name == True.

    Returns:
        ood_category_dict (dict):  The dictionary as described above.
    """
    w2v = downloader.load("word2vec-google-news-300")
    # Get dict of entities for each domain + substitute
    domains = list(domain_entity_dict.keys())
    entity_dict = {}
    for domain in domains:
        entities = set()
        for entity in domain_entity_dict[domain]:
            if substitutions is not None and entity in substitutions:
                entities.add(substitutions[entity])
            else:
                entities.add(entity)
        entity_dict[domain] = sorted(list(entities))
    
    # Do a clustering for each OOD possibility
    ood_category_dict = {}
    for test_domain in domains:
        # List of all entities not in the test domain
        train_entities = set()
        [train_entities.update(domain_entities) for domain, domain_entities in entity_dict.items() if domain != test_domain]
        train_entities = sorted(list(train_entities))
        # And for those only in the test domain
        test_entities = [entity for entity in entity_dict[test_domain] if entity not in train_entities]
        # Get embeddings for them
        train_embeddings = w2v[train_entities]
        # Only if there are entitites unique to the test domain
        if len(test_entities) > 0:
            test_embeddings = w2v[test_entities]

        # Fit the pipe
        n_components = min(len(train_entities), n_components)
        pipe = Pipeline([
            ("PCA", PCA(n_components=n_components, svd_solver="full")),
            ("UMAP", UMAP(n_neighbors=n_neighbors, n_components=3, metric="cosine", min_dist=min_dist, random_state=random_state)),
            ("Clusterer", AffinityPropagation(damping=damping, random_state=random_state))
        ])
        train_pred = pipe.fit_predict(train_embeddings)
        # Predict cluster of test domain entitites (if there are any)
        if len(test_entities) > 0:
            test_pred = pipe.predict(test_embeddings)
            # Concat everything and continue like before
            pred = list(train_pred) + list(test_pred)
            entities = train_entities + test_entities
        else:
            pred = train_pred
            entities = train_entities
        categories = categories_from_pred(pred, entities)
        if name:
            categories = name_categories(categories, w2v, stop_chars=stop_chars, kth_closest=kth_closest)
        ood_category_dict[test_domain] = entities_to_orig(categories, substitutions)
    return ood_category_dict

def get_synsets() -> tuple:
    """Get synsets, entities, and substitutions for the WordNet based
    clusterers.
    """
    entities_orig = ["academicjournal", "album", "algorithm", "astronomicalobject", "award", "band", "book", "chemicalcompound", "chemicalelement", "conference", "country", "discipline", "election", "enzyme", "event", "field", "literarygenre", "location", "magazine", "metrics", "misc", "musicalartist", "musicalinstrument", "musicgenre", "organisation", "person", "poem", "politicalparty", "politician", "product", "programlang", "protein", "researcher", "scientist", "song", "task", "theory", "university", "writer"]
    substitutions = {"musicalartist": "musician", "organisation": "organization", "politicalparty": "party", "academicjournal": "journal", "chemicalcompound": "chemical", "chemicalelement": "chemical", "astronomicalobject": "galaxy", "musicgenre": "genre", "literarygenre": "genre", "programlang": "java", "musicalinstrument": "instrument", "misc": "miscellaneous"}
    entities = set()
    for entity in entities_orig:
        if entity in substitutions:
            entities.add(substitutions[entity])
        else:
            entities.add(entity)

    entities = sorted(list(entities))
    check = ['album', 'algorithm', 'award', 'band', 'book', 'chemical', 'conference', 'country', 'discipline', 'election', 'enzyme', 'event', 'field', 'galaxy', 'genre', 'instrument', 'java', 'journal', 'location', 'magazine', 'metrics', 'miscellaneous', 'musician', 'organization', 'party', 'person', 'poem', 'politician', 'product', 'protein', 'researcher', 'scientist', 'song', 'task', 'theory', 'university', 'writer']
    if entities != check:
        raise ValueError("entities modified since meaning_dict was edited!")
    meaning_dict = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 3, 13: 2, 14: 0, 15: 5, 16: 2, 17: 1, 18: 0, 19: 0, 20: 3, 21: 0, 22: 0, 23: 2, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 1, 34: 0, 35: 2, 36: 0}
    # Get categories with associated words
    entity_syns = []
    for i, word in enumerate(entities):
        entity_syns.append(wordnet.synsets(word)[meaning_dict[i]])
    return entity_syns, entities, substitutions

def get_topological(level: int = 2) -> dict:
    """Get topological categories based on WordNet.  Entities,
    substitutions, and the word form -> word sense (synset)
    specification dict is hard-coded to avoid mismatches.

    Parameters:
        level (int, optional):  Which dept of synsets to categorize the
            words at (0 is the root).  Defaults to 2.
    
    Retruns:
        topological_categories (dict):  The category dictionary in
            category (key): list of entities (value) format.  The
            substituted entities are returned to the original form, even
            if there were several that mapped to the same.
    """
    entity_syns, entities, substitutions = get_synsets()
    topological_categories = topological_category_dict(entity_syns, level, entities)
    topological_categories = entities_to_orig(topological_categories, substitutions)
    return topological_categories

def get_thesaurus_affinity_based(
        random_state: Union[int, None] = None,
        damping: float = 0.5,
        name: bool = True,
        stop_chars: Union[str, list, None] = "._@",
        kth_closest: int = 1
    ) -> dict:
    """Get category dictionary based on WordNet similarities, with
    AffinityPropagation clusterer.

    Parameters:
        random_state (int|None, optional):  Seed for probabilistic
            components.  Default is None, no seed.
        damping (float, optional):  Arg for AffinityPropagation,
            defaults to 0.5
        name (bool, optional):  Name categories based on closest word
            in embedding space to mean of cluster?  Default: True.
        stop_chars (str|list|None, optional):  If closest word contains
            any of these, use next closest.  Only takes effect if
            name == True.  If str, Each char in it will be individually
            considered.  If list, list elements as a whole will be
            considered.  None signal no stop_chars.  Default: '._@'.
        kth_closest (int, optional):  Determines which word is selected
            from the list of most similar ones.  Must be in a positive
            int but less than the number of words in the model.
            Defaults to 1 (the most similar one). Only takes effect if
            name == True.

    Returns:
        thesaurus_affinity_categories (dict):  The category dictionary 
        in category (key): list of entities (value) format.  The
        substituted entities are returned to the original form, even if
        there were several that mapped to the same.
    """
    entity_syns, entities, substitutions = get_synsets()
    # Calculate Wu-Palmer Similarities
    similarities = get_wup_sim(entity_syns)
    # Affinity Propagation
    ap = AffinityPropagation(damping=damping, affinity="precomputed", random_state=random_state)
    ap = ap.fit_predict(similarities)

    # Category names
    thesaurus_affinity_categories = categories_from_pred(ap, entities)
    if name:
        w2v = downloader.load("word2vec-google-news-300")
        thesaurus_affinity_categories = name_categories(thesaurus_affinity_categories, w2v, stop_chars=stop_chars, kth_closest=kth_closest)
    thesaurus_affinity_categories = entities_to_orig(thesaurus_affinity_categories, substitutions)
    return thesaurus_affinity_categories

def get_entity2category(category2entities: dict) -> dict:
    """Get a dictionary in entity (key, str): category (value, str)
    format from a dict in category (key, str): entities (value, list of
    str) format.

    Parameters:
        category2entities (dict):  Dict to convert.
    
    Returns:
        entity2category (dict):  Resulting dict.
    """
    entity2category = {}
    for category, entities in category2entities.items():
        for entity in entities:
            if entity in entity2category:
                raise ValueError(f"{entity} is in the dict several times.")
            entity2category[entity] = category
    return entity2category

def get_to_remove(unique_entities: dict) -> dict:
    """Get dict with domains are keys and values are the set of entities
    that domain cannot contain, i.e. the set of all unique entities of
    all other domains.

    Parameters:
        unique_entities (dict):  Dict with domains as keys and list of
            entities unique to that domain as values.

    Returns:
        to_remove (dict):  As specified above.
    """
    to_remove = {}
    for from_domain_remove in unique_entities.keys():
        to_remove[from_domain_remove] = set()
        for domain_unique, entities in unique_entities.items():
            if domain_unique != from_domain_remove:
                to_remove[from_domain_remove].update(entities)
    return to_remove

def get_ood_cluster_domain_entity_dict(domain_entity_dict: dict, to_remove: dict) -> dict:
    """Get dictionary for ood entity clustering.

    Parameters:
        domain_entity_dict (dict):  Keys are domains, values are the
            list of entitites that occur in the corpus for that domain.
        to_remove (dict):   Keys are domains, values are the set of
            entities that domain cannot contain.
    
        Returns:
            ood_dict (dict):  Keys are domains, values are entities that
                will occur in the corpus for given domain after removing
                the unique entities of other domains.
    """
    ood_dict = {}
    for domain, entities in domain_entity_dict.items():
        ood_dict[domain] = []
        for entity in entities:
            if entity not in to_remove[domain]:
                ood_dict[domain].append(entity)
        ood_dict[domain].sort()
    return ood_dict

def create_ood_clustering_data(unique_entities: dict, data_path: str, destination_path: str) -> None:
    """Create dataset where the entities listed in `unique_entities` are
    unique to the matching domain.  Removes all sentences where
    a unique entity type is in another domain, and writes the removed
    counts per domain and per train/dev/test set to file.  Randomly
    selected sentences are moved from the dev set to the train and test
    sets to replenish their original size.  All domains partaking in the
    training must be the `unique_entities` dict, even if they don't have
    any unique entities, in which case the value should be an empty
    list.

    Parameters:
        unique_entities (dict):  domain (key, str): unique entities
            (value, list of str) pairs.
        data_path (str):  Path to original data folder.
        destination_path (str):  Folder to save new data in.

    Returns:
        None
    """
    # Check if all entities are only unique to only one domain
    unique_list =[]
    unique_set = set()
    for entities in unique_entities.values():
        unique_list += entities
        unique_set.update(entities)
    if len(unique_set) != len(unique_list):
        raise ValueError("An entity can only be unique to one domain.")

    os.makedirs(destination_path, exist_ok=True)
    # For each domain create dict what to remove
    to_remove = get_to_remove(unique_entities)    

    removed_per_domain = {}
    removed_count = {}
    for domain, entities_to_remove in to_remove.items():
        removed_count = {}  # technically not necessary but easier to read
        for dataset in ["train", "test", "dev"]:
            to_write = []
            removed_count[dataset] = 0
            kept_line_count = 0  # Only matter for dev
            with open(f"{data_path}{domain}-{dataset}.json") as from_file:
                for json_elem in from_file:
                    document = json.loads(json_elem)
                    # If any entity in to remove list for given domain for given line...
                    for ner in document["ner"]:
                        if ner[2] in entities_to_remove:
                            removed_count[dataset] += 1
                            break
                    # ... break the inner for loop -> 'else' won't execute (for-else construct)
                    else:
                        # if loop wasn't broken -> add line to buffer
                        kept_line_count += 1
                        to_write.append(json_elem)

            if dataset != "dev":
                with open(f"{destination_path}{domain}-{dataset}.json", "w") as to_file:
                    for json_elem in to_write:
                        to_file.write(json_elem)
            else:
                nr_to_remove = removed_count["train"] + removed_count["test"]
                # Randomly choose the sentences that will be reallocated to train/test
                idx_to_remove = np.random.randint(0, kept_line_count, nr_to_remove)
                # We can just append them to the end of the train / test file, no random insertion needed bc shuffling dataloader
                with open(f"{destination_path}{domain}-train.json", "a") as to_file:
                    for idx in idx_to_remove[:removed_count["train"]]:
                        to_file.write(to_write[idx])
                with open(f"{destination_path}{domain}-test.json", "a") as to_file:
                    for idx in idx_to_remove[removed_count["train"]:]:
                        to_file.write(to_write[idx])
                to_write = [line for i, line in enumerate(to_write) if i not in idx_to_remove]
                with open(f"{destination_path}{domain}-{dataset}.json", "w") as to_file:
                    for json_elem in to_write:
                        to_file.write(json_elem)
        removed_per_domain[domain] = removed_count
    with open(f"{destination_path}removed_per_domain.json", "w") as f:
        json.dump(removed_per_domain, f, indent=4)

def get_categories(
        mapping_type: str,
        domains: list = ["ai", "literature", "music", "news", "politics", "science"],
        mapper_params: Union[dict, None] = None,
        incl_category2entity: bool = False,
        unique_entities: Union[dict, None] = None
    ) -> Union[dict, tuple]:
    """Get categorization of named entities as dictionary based on
    different kind of clustering methods (mapping_type).  Options:
    ['manual', 'elisa', 'embedding', 'ood_clustering', 'topological',
    'thesaurus_affinity'].  The named entity labels for the list of
    domains provided are read from the .env file.  The optional args
    for the different mapping functions can be passed as the
    mapper_params arg as a dict.  For 'ood_clustering' `unique_entities`
    must be passed and the set of its keys must match the list of
    `domains`.

    The returned dictionary will be in the format entity (key, str):
    category (value, str), if incl_category2entity == True then a second
    dict with in in category (key, str): entities (value, list of str)
    format will also be returned.  For the ood_clustering mapping type
    either of these a wrapper dict will be returned which has the test
    domains as keys and the corresponding mapping dicts as values.

    Parameters:
        mapping_type (str):  As described above.
        domains (list, optional):  List of domains to be included.
            Defaults to include all 6.  Only takes effect if
            mapping_type is in ['embedding', 'ood_clustering'], else all
            domains are considered.
        mapper_params (dict|None, optional):  (Optional) args to pass to
            specific mapper functions, see their docstrings for options.
            Use str matching the arg name as key and the desired value
            as well, you guessed it, value.
        incl_category2entity (bool, optional):  If True, a second dict
            will also be returned, mapping categories to entities.
            Default: False.
        unique_entities (dict|None, optional):  Keys are domains, values
            are entities unique to the domain.  Only takes effect if
            `mapping_type` == 'ood_clustering'.  If given, all domains
            partaking in the training must be in it, even if they don't
            have any unique entities, in which case the value should be
            an empty list.  Defaults to None.
    
    Returns:
        entity2category (dict):  Mapping dictionary as described above,
            entity to category.
        category2entity (dict, optional):  Mapping dictionary as
            described above, category to entities.  Only returned if
            `incl_category2entity` == True.
    """
    if mapping_type not in ["manual", "elisa", "embedding", "ood_clustering", "topological", "thesaurus_affinity"]:
        raise ValueError('Mapping_type must be in ["manual", "elisa", "embedding", "ood_clustering", "topological", "thesaurus_affinity"]')

    # Empty dict is ok for unpacking kwargs in function calls but None is not
    if mapper_params is None:
        mapper_params = {}
    
    if mapping_type in ["manual", "elisa"]:
        if mapping_type == "manual":
            mapping = get_manual()
        else:
            mapping = get_elisa()
        if incl_category2entity:
            return get_entity2category(mapping), mapping
        else:
            return get_entity2category(mapping)

    if mapping_type in ["embedding", "ood_clustering"]:
        substitutions = {"musicalartist": "musician", "organisation": "organization", "politicalparty": "coalition", "academicjournal": "journal", "chemicalcompound": "chemical", "chemicalelement": "chemical", "astronomicalobject": "galaxy", "musicgenre": "genre", "literarygenre": "genre", "programlang": "javascript", "musicalinstrument": "violin", "misc": "miscellaneous"}
    # Not currently used, the thesaurus based stuff is hard-coded because the
    # meaning dict ordering would be really hard to implement otherwise.
    else:
            substitutions = {"musicalartist": "musician", "organisation": "organization", "politicalparty": "party", "academicjournal": "journal", "chemicalcompound": "chemical", "chemicalelement": "chemical", "astronomicalobject": "galaxy", "musicgenre": "genre", "literarygenre": "genre", "programlang": "java", "musicalinstrument": "instrument", "misc": "miscellaneous"}

    if mapping_type == "ood_clustering" and unique_entities is None:
        raise ValueError("For 'ood_clustering' `mapping_type` unique_entities dict must be passed.")
    
    if mapping_type == "ood_clustering" and set(domains) != set(unique_entities.keys()):
        raise ValueError("For 'ood_clustering' `mapping_type` the keys of `unique_entities` must match set(`domains`).")

    domain_entity_dict = {}
    all_entities = set()
    for domain in domains:
        domain_entity_dict[domain] = sorted(os.getenv(f"{domain.upper()}_LABELS").split())
        all_entities.update(os.getenv(f"{domain.upper()}_LABELS").split())
    # All as in all within the domains specified
    all_entities = sorted(list(all_entities))

    # This is a test if i messed up the code above
    if len(domains) == 6 and all_entities != sorted(os.getenv("ENTITY_LABELS").split()):
        raise ValueError("All domains are given but entity list wasn't built successfully.")
    
    if mapping_type == "ood_clustering":
        to_remove = get_to_remove(unique_entities)
        domain_entity_dict = get_ood_cluster_domain_entity_dict(domain_entity_dict, to_remove)
        mapping =  get_ood_embedding_based(domain_entity_dict, substitutions, **mapper_params)
    elif mapping_type == "embedding":
        mapping =  get_embedding_based(all_entities, substitutions, **mapper_params)
    elif mapping_type == "topological":
        mapping =  get_topological(**mapper_params)
    else:
        mapping =  get_thesaurus_affinity_based(**mapper_params)
    
    if mapping_type != "ood_clustering":
        if incl_category2entity:
            return get_entity2category(mapping), mapping
        else:
            return get_entity2category(mapping)
    
    # If ood_clustering
    entity2category_dict = {}
    for domain, map in mapping.items():
        entity2category_dict[domain] = get_entity2category(map)
    if incl_category2entity:
        return entity2category_dict, mapping
    else:
        return entity2category_dict

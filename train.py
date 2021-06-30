import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from models.BayesianRidgeRegression import BayesianRidgeRegression
from models.LogisticRegression import LogisticRegression
from models.RandomForestClassifier import RandomForestClassifier
from models.SupportVectorClassifier import SupportVectorClassifier
from mutators.PermutationalMutator import PermutationalMutator
from mutators.GeneticMutator import GeneticMutator
import logging
from alive_progress import alive_bar

default_models = [
    BayesianRidgeRegression,
    LogisticRegression,
    RandomForestClassifier,
    SupportVectorClassifier
]

default_mutators = [
    PermutationalMutator,
    GeneticMutator,
]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train(sdb, models=default_models, mutators=default_mutators, chunksize=64, pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
    """Train models on given PDB data"""
    pdbs = sdb.get_pdbs()
    f = open(pdb_data_file, 'rb')
    pdb_data = pickle.load(f)
    f.close()
    f = open(pdb_ids_file, 'rb')
    pdb_ids = pickle.load(f)
    f.close()

    total_iterations = len(pdb_ids) ** 2
    logging.debug("Total iterations: %s", total_iterations)

    sublists = list(chunks(pdbs, chunksize))

    max_length = 0
    for model_type in models:
        logging.info("Training models of type: %s", model_type.__name__)
        for mutator_type in mutators:
            mutator = mutator_type(model_type)
            logging.info("Using %s", mutator.get_name())
            logging.debug("Estimated length: %s", mutator.get_estimated_total())
            with alive_bar(mutator.get_estimated_total(), 'Training') as bar:
                while mutator.has_next():
                    m = mutator.next()
                    logging.debug("[%s] Created %s with options: %s", mutator, m, mutator.get_current_options())
                    # i = 0
                    for sublist in sublists:
                        logging.debug("Running on: %s", sublist)
                        # print("PROGRESS: {}%, {} / {}".format(i*100*chunksize/len(pdbs), i*chunksize, len(pdbs)))

                        max_length = max(_run_training(sdb, sublist, m, pdb_ids, pdb_data), max_length)
                        # i += 1

                    logging.debug("Accuracy of %s: %s", m.get_full_name(), m.get_accuracy())
                    m.save()
                    bar()

    logging.debug("Max length: %s", max_length)

    return max_length

def _run_training(sdb, sublist, m, pdb_ids, pdb_data):
    """Perform one round of training on the models using a given sublist of data"""
    logging.debug("Reading scores")
    x = pd.DataFrame()
    y = []
    max_length = 0
    for pdb1 in sublist:
        for pdb2 in sublist:
            if pdb1 in pdb_ids and pdb2 in pdb_ids:
                if pdb_ids[pdb1] is not None and pdb_ids[pdb2] is not None:
                    id1 = pdb_ids[pdb1]
                    data1 = pdb_data[pdb1]
                    id2 = pdb_ids[pdb2]
                    data2 = pdb_data[pdb2]
                    arr = [np.concatenate((data1, data2), axis=None).tolist()]
                    max_length = max(len(arr[0]), max_length)
                    x = x.append(arr)
                    score_class = sdb.score_mapped(id1, id2)[0]
                    y.append(score_class)

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    x = x.to_numpy()
    x = imp.fit_transform(x)
    y = np.array(y)
    logging.debug("Added %s rows.", len(x))

    logging.debug("Training %s...", m.get_full_name())
    try:
        m.fit(x, y)
    except ValueError:
        logging.warning("Only one class found, skipping...")
    except:
        logging.error("Unexpected error: %s", e)

    return max_length

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
    print("Total iterations: {}".format(total_iterations))

    sublists = list(chunks(pdbs, chunksize))

    model_objs = {}
    max_length = 0
    for model_type in models:
        for mutator_type in mutators:
            mutator = mutator_type(model_type)
            model_objs[mutator.get_name()] = []
            while mutator.hasNext():
                m = mutator.next()
                model_objs[mutator.get_name()].append({'options': mutator.get_current_options(), 'model': m})
                print("[{!s}]: Created {!s} with options:".format(mutator, m))
                print(mutator.get_current_options())
                i = 0
                for sublist in sublists:
                    print("------------------------------")
                    print("Running on: {}".format(sublist))
                    print("------------------------------")
                    print("PROGRESS: {}%, {} / {}".format(i*100*chunksize/len(pdbs), i*chunksize, len(pdbs)))

                    max_length = max(_run_training(sdb, sublist, m, pdb_ids, pdb_data), max_length)
                    i += 1

    print("Max length: {}", max_length)

    print("--- SAVING ---")

    for m in model_objs:
        print("Saving {!s}...".format(m))
        m.save()

    print("--- ACCURACY CHECKS ---")

    for m in model_objs:
        print("{!s}: {}".format(m, m.get_accuracy()))

    print("Done training.")
    return (model_objs, max_length)

def _run_training(sdb, sublist, models, pdb_ids, pdb_data):
    """Perform one round of training on the models using a given sublist of data"""
    print("--- READING SCORES ---")
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
    print("Added {} rows.".format(len(x)))

    print("--- TRAINING ---")

    print("Training {!s}... ".format(m), end='')
    try:
        m.fit(x, y)
        print("No errors.")
    except ValueError:
        print("Warning: Only one class found, skipping...")
    except:
        print("Unexpected error: {!s}".format(e))

    return max_length

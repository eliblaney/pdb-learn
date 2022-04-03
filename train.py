import os
import config
import strings
import logging
import joblib
import numpy as np
from models.BayesianRidgeRegression import BayesianRidgeRegression
from models.LogisticRegression import LogisticRegression
from models.RandomForestClassifier import RandomForestClassifier
from models.SupportVectorClassifier import SupportVectorClassifier
from mutators.PermutationalMutator import PermutationalMutator
from mutators.GeneticMutator import GeneticMutator
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

def train_partitioned(sdb, folder='data', models=default_models, mutators=default_mutators):
    # Input and output files should be similarly named, with 'input' or 'output' present
    inputs = [file for file in os.listdir(folder) if 'input' in file]
    outputs = [x.replace('input', 'output') for x in inputs]

    for mt in default_models:
        mutator = PermutationalMutator(mt)
        while mutator.has_next():
            m = mutator.next()
            logging.debug("[%s] Created %s with options: %s", mutator, m, mutator.get_current_options())
            m.save()

    models = [file for file in os.listdir('saved_models')]
    len_models = len(models)

    total_partitions = len(inputs)
    i = 0
    with alive_bar(total_partitions, 'Partition') as bar:
        while i < total_partitions:
            x = np.load(folder + '/' + inputs[i], allow_pickle=True)
            y = np.load(folder + '/' + outputs[i], allow_pickle=True)
            
            with alive_bar(len_models, 'Models') as bar2:
                for mp in models:
                    m = load_model('saved_models/' + mp)

                    logging.debug("Training %s on partition %s", m.get_full_name(), str(i))
                    try:
                        m.partial_fit(x, y)
                    except ValueError:
                        logging.warning("Only one class found, skipping...")
                    except Exception as e:
                        logging.error("Unexpected exception: %s", e)

                    logging.debug("Accuracy of %s: %s", m.get_full_name(), m.get_accuracy())
                    m.save()
                    bar2()

            # Free up memory
            del x
            del y
            bar()
            i = i + 1

def train(sdb, models=default_models, mutators=default_mutators, inputs_file='inputs.np', outputs_file='outputs.np'):
    """Train models on given PDB data"""
    x = np.load(inputs_file)
    y = np.load(outputs_file)

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

                    logging.debug("Training %s...", m.get_full_name())
                    try:
                        m.fit(x, y)
                    except ValueError:
                        logging.warning("Only one class found, skipping...")
                    except Exception as e:
                        logging.error("Unexpected exception: %s", e)
        
                    logging.debug("Accuracy of %s: %s", m.get_full_name(), m.get_accuracy())
                    m.save()
                    bar()

def load_model(path):
    if not os.path.exists(path):
        return None

    name = os.path.basename(path).split(".")[0]
    model = joblib.load(path)
    m = None
    if name == 'bayesian_ridge_regression':
        m = BayesianRidgeRegression(None)
    elif name == 'logistic_regression':
        m = LogisticRegression(None)
    elif name == 'random_forest_classifier':
        m = RandomForestClassifier(None)
    elif name == 'support_vector_classifier':
        m = SupportVectorClassifier(None)
    else:
        return None

    m.model = model
    return m

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    train_partitioned(sdb)

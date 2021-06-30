import config
import strings
import logging
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train(sdb, models=default_models, mutators=default_mutators, chunksize=64, inputs_file='inputs.np', outputs_file='outputs.np'):
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
                    except:
                        logging.error("Unexpected error: %s", e)
        
                    logging.debug("Accuracy of %s: %s", m.get_full_name(), m.get_accuracy())
                    m.save()
                    bar()

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    train(sdb)

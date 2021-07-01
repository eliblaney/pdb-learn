import os
import random
import joblib
import strings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from models.BayesianRidgeRegression import BayesianRidgeRegression
from models.LogisticRegression import LogisticRegression 
from models.RandomForestClassifier import RandomForestClassifier
from models.SupportVectorClassifier import SupportVectorClassifier
import pdb
import logging
import config

def rand_pdb():
    """Select a random PDB entry from the set"""
    p = random.choice(pdbs)
    pdb_data = pdb.get_pdb_data(p, strings.pdbbind(p, general=True))
    strings_id = strings.strings_id(p)
    return (pdb_data, strings_id)

def predict(sdb, num=1, each=1000, prediction_file='predictions.csv', results_file='results.csv', max_length=3720, model_dir='saved_models/'):
    """Run `num` rounds of `each` predictions on the models"""
    pdbs = sdb.get_pdbs(general=True)

    logging.debug('Running %s prediction rounds of %s each', num, each)
    logging.debug("Total predictions: %s", num * each)

    models = os.listdir(model_dir)

    predictions = pd.DataFrame(columns=['model','predicted_y','actual_y','correct'])
    results = pd.DataFrame(columns=['model','accuracy'])
    for i in range(num):
        logging.debug('----------------------------')
        logging.debug('Round: {} [of {}]', i + 1, num)
        (predictions, results) = _run_predictions(predictions, results, models, max_length, each)

    logging.debug('----------------------------')
    logging.debug('Saving predictions...')
    predictions.to_csv(prediction_file, index=False)
    results.to_csv(results_file, index=False)

    return (prediction_file, results_file)

def _run_predictions(predictions, results, models, max_len=3720, num=100):
    """Run num predictions on the models"""
    logging.debug("Max length: %s", max_len)
    xs = pd.DataFrame(columns=np.arange(0,max_len))
    ys = []
    i = num
    while i > 0:
        (p1d, p1i) = rand_pdb()
        (p2d, p2i) = rand_pdb()
        if p1i is not None and p2i is not None:
            actualscore = strings.score_mapped(p1i, p2i)[0]
            if actualscore != -1: # Filter for combinations that have existing scores
                ys.append(actualscore)
                arr = [np.concatenate((p1d, p2d), axis=None).tolist()]
                if len(arr[0]) <= max_len:
                    xs = xs.append(arr)
                    i -= 1
                    if i % 25 == 0 and i > 0:
                        logging.debug("%s remaining...", i)

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    xs = imp.fit_transform(xs.to_numpy())
    ys = np.array(ys)
    logging.debug("Added %s rows.", len(xs))
    logging.debug("Shape: %s", xs.shape)

    logging.info("Running predictions")

    model_scores = {}

    with alive_bar(len(models), title='Predicting') as bar:
        for m in models:
            m = load_model(m)
            name = m.get_full_name()
            model_scores[name] = 0
            for i in range(len(xs)):
                predictedy = m.predict([xs[i]])[0]
                actualy = ys[i]
                correct = actualy == round(predictedy)

                predictions = predictions.append([{'model': name, 'predicted_y': predictedy, 'actual_y': actualy, 'correct': 1 if correct else 0}])

                if correct:
                    model_scores[name] += 1

            model_scores[name] /= (len(ys)/100.0)
            bar()

    logging.debug("--- RESULTS ---")
    for m in model_scores.keys():
        logging.debug("{}: {} percent".format(m, model_scores[m]))
        results = results.append([{'model': str(m), 'accuracy': model_scores[m]}])

    return (predictions, results)

def load_model(path):
    if not os.path.exists(path):
        return None

    name = os.path.basename(path).split("\.")[0]
    model = joblib.load(path)
    m = None
    if name == 'Bayesian Ridge Regression':
        m = BayesianRidgeRegression(None)
    elif name == 'Logistic Regression':
        m = LogisticRegression(None)
    elif name == 'Random Forest Classifier':
        m = RandomForestClassifier(None)
    elif name == 'Support Vector Classifier':
        m = SupportVectorClassifier(None)
    else:
        return None

    m.model = model
    return m

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    predict(sdb)

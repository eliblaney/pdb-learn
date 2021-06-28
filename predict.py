import random
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from models.BayesianRidgeRegression import BayesianRidgeRegression
from models.LogisticRegression import LogisticRegression
from models.RandomForestClassifier import RandomForestClassifier
from models.SupportVectorClassifier import SupportVectorClassifier
import pdb

def rand_pdb():
    """Select a random PDB entry from the set"""
    p = random.choice(pdbs)
    pdb_data = pdb.get_pdb_data(p, strings.pdbbind(p, general=True))
    strings_id = strings.strings_id(p)
    return (pdb_data, strings_id)

def _run_predictions(predictions, results, models, max_len=3720, num=100):
    """Run num predictions on the models"""
    print("Max length: {}".format(max_len))
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
                        print("{} remaining...".format(i))

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    xs = imp.fit_transform(xs.to_numpy())
    ys = np.array(ys)
    print("Added {} rows.".format(len(xs)))
    print("Shape: {}".format(xs.shape))

    print("--- PREDICTING ---")

    model_scores = {}

    for m in models:
        model_scores[str(m)] = 0
        for i in range(len(xs)):
            predictedy = m.predict([xs[i]])[0]
            actualy = ys[i]
            correct = actualy == round(predictedy)

            predictions = predictions.append([{'model': str(m), 'predicted_y': predictedy, 'actual_y': actualy, 'correct': 1 if correct else 0}])

            if correct:
                model_scores[str(m)] += 1

        model_scores[str(m)] /= (len(ys)/100.0)

    print("--- RESULTS ---")
    for m in model_scores.keys():
        print("{}: {} percent".format(m, model_scores[m]))
        results = results.append([{'model': str(m), 'accuracy': model_scores[m]}])

    return (predictions, results)

def predict(sdb, num=100, each=100, prediction_file='predictions.csv', results_file='results.csv', max_length=3720, models=None):
    """Run `num` rounds of `each` predictions on the models"""
    print("Reading PDB data...")
    pdbs = sdb.get_pdbs(general=True)

    print('Running {} prediction rounds of {} each'.format(num, each))
    print("Total predictions: {}".format(num * each))

    if not models:
        print("Loading models...")
        models = [
            BayesianRidgeRegression(),
            LogisticRegression(),
            RandomForestClassifier(),
            SupportVectorClassifier()
        ]
        for m in default_models:
            print("Loading {!s}...".format(m))
            m.load()

    predictions = pd.DataFrame(columns=['model','predicted_y','actual_y','correct'])
    results = pd.DataFrame(columns=['model','accuracy'])
    for i in range(num):
        print('----------------------------')
        print('Round: {} [of {}]'.format(i + 1, num))
        (predictions, results) = _run_predictions(predictions, results, models, max_length, each)

    print('----------------------------')
    print('Saving predictions...')
    predictions.to_csv(prediction_file, index=False)
    results.to_csv(results_file, index=False)

    print('Finished predicting.')
    return (prediction_file, results_file)

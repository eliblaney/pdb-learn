import config
import pickle
import numpy as np
import pandas as pd
import logging
import strings
from sklearn.impute import SimpleImputer
from alive_progress import alive_bar

def build(sdb):
    pdbs = sdb.get_pdbs()
    f = open(pdb_data_file, 'rb')
    pdb_data = pickle.load(f)
    f.close()
    f = open(pdb_ids_file, 'rb')
    pdb_ids = pickle.load(f)
    f.close()

    x = pd.DataFrame()
    y = []
    max_length = 0

    total_iterations = len(pdbs) ** 2
    logging.debug("Total iterations: %s", total_iterations)

    with alive_bar(total_iterations)**2) as bar:
        for pdb1 in pdbs:
            for pdb2 in pdbs:
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
                bar()

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    x = x.to_numpy()
    x = imp.fit_transform(x)
    y = np.array(y)

    logging.debug("Input shape: %s", x.shape)
    logging.debug("Output shape: %s", y.shape)
    logging.debug("Max length: %s", max_length)

    np.save('inputs.np', x)
    np.save('outputs.np', y)

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    build(sdb)

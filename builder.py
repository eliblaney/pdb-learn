import config
import pickle
import numpy as np
import pandas as pd
import logging
import strings
import _thread
from sklearn.impute import SimpleImputer
from alive_progress import alive_bar

class PDBBuilder:

    sdb = None
    x = []
    y = []
    max_length = 0
    pdb_ids = None
    pdb_data = None

    def __init__(self, sdb, pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
        self.sdb = sdb

        f = open(pdb_data_file, 'rb')
        self.pdb_data = pickle.load(f)
        f.close()
        f = open(pdb_ids_file, 'rb')
        self.pdb_ids = pickle.load(f)
        f.close()

    def build(self, cpus=8):
        pdbs = [pdb for pdb in list(self.pdb_ids.keys()) if self.pdb_ids[pdb]] # Get PDB IDs that match StringDB IDs
        pdb_chunks = self.chunks(pdbs, cpus)

        self.x = []
        self.y = []
        self.max_length = 0

        total_iterations = len(pdbs) ** 2
        logging.debug("Total iterations: %s", total_iterations)

        logging.info("Starting %s threads", cpus)
        with alive_bar(total_iterations) as bar:
            for chunk in pdb_chunks:
                _thread.start_new_thread(self._build, (chunk, bar))                

        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        self.x = np.array(self.x)
        self.x = imp.fit_transform(self.x)
        self.y = np.array(self.y)

        logging.debug("Input shape: %s", self.x.shape)
        logging.debug("Output shape: %s", self.y.shape)
        logging.debug("Max length: %s", self.max_length)

        np.save('inputs.np', self.x)
        np.save('outputs.np', self.y)

    def _build(self, chunk, bar):
        for pdb1 in chunk:
            for pdb2 in chunk:
                id1 = self.pdb_ids[pdb1]
                data1 = self.pdb_data[pdb1]
                id2 = self.pdb_ids[pdb2]
                data2 = self.pdb_data[pdb2]
                arr = [np.concatenate((data1, data2), axis=None).tolist()]
                self.max_length = max(len(arr[0]), self.max_length)
                self.x.append(arr)
                score_class = self.sdb.score_mapped(id1, id2)[0]
                self.y.append(score_class)
                bar()

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    b = PDBBuilder(sdb)
    b.build()

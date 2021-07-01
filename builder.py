import config
import pickle
import numpy as np
import pandas as pd
import logging
import strings
from threading import Thread
from sklearn.impute import SimpleImputer
from alive_progress import alive_bar

class PDBBuilder:

    sdb = None
    x = []
    y = []
    pdb_ids = None
    pdb_data = None
    pdbs = None

    def __init__(self, sdb, pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
        self.sdb = sdb

        f = open(pdb_data_file, 'rb')
        self.pdb_data = pickle.load(f)
        f.close()
        f = open(pdb_ids_file, 'rb')
        self.pdb_ids = pickle.load(f)
        f.close()

        self.pdbs = [pdb for pdb in list(self.pdb_ids.keys()) if self.pdb_ids[pdb]] # Get PDB IDs that match StringDB IDs

    
    def partition(self, num_partitions, cpus=8):
        pdbs = self.chunks(pdbs, num_partitions)
        for i in range(num_partitions):
            logging.info("Buliding partition %s", i)
            self.build(pdbs=pdbs, inputfile='input_' + i + '.np', outputfile='output_' + i + '.np')

    def build(self, pdbs=None, cpus=8, inputfile='inputs.np' outputfile='outputs.np'):
        if pdbs is None:
            pdbs = self.pdbs
        pdb_chunks = self.chunks(pdbs, cpus)

        self.x = []
        self.y = []

        total_iterations = len(pdbs) ** 2
        logging.debug("Total iterations: %s", total_iterations)

        logging.info("Starting %s threads", cpus)
        threads = []
        with alive_bar(total_iterations) as bar:
            for chunk in pdb_chunks:
                t = Thread(target=self._build, args=(chunk, bar))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        self.x = pd.DataFrame(self.x).to_numpy()
        self.x = imp.fit_transform(self.x)
        self.y = np.array(self.y)

        logging.debug("Input shape: %s", self.x.shape)
        logging.debug("Output shape: %s", self.y.shape)

        folder = 'data'
        np.save(folder + '/' + inputfile, self.x)
        np.save(folder + '/' + outputfile, self.y)

    def _build(self, chunk, bar):
        for pdb1 in chunk:
            for pdb2 in chunk:
                id1 = self.pdb_ids[pdb1]
                data1 = self.pdb_data[pdb1]
                id2 = self.pdb_ids[pdb2]
                data2 = self.pdb_data[pdb2]
                arr = [np.concatenate((data1, data2), axis=None).tolist()]
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
    b.partition(30)

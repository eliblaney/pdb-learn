import config
import strings
import numpy as np
import pandas as pd
import pickle
import pdb
import logging
from alive_progress import alive_bar
import rpdb

def create(sdb, pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
    LIMIT = -1
    pdbs = sdb.get_pdbs()
    total = LIMIT if LIMIT != -1 else len(pdbs)
    total = 2*total+3

    with alive_bar(title='Parsing PDBs', total=total) as bar:
        bar.text('Building dictionaries')
        pdb_data = {}
        pdb_ids = {}
        longest_num_residues = -1
        for p in pdbs:
            d = pdb.get_pdb_data(p, sdb.pdbbind(p))
            num_residues = len(d)
            if num_residues > longest_num_residues:
                longest_num_residues = num_residues

            pdb_data[p] = d
            pdb_ids[p] = sdb.strings_id(p)
            bar()
            if LIMIT != -1:
                LIMIT = LIMIT - 1
                if LIMIT == 0: break

        longest_residue_length = -1
        for d in pdb_data.values():
            for residue in d:
                residue_length = len(residue)
                if residue_length > longest_residue_length:
                    longest_residue_length = residue_length
            bar()

        logging.debug("Longest number of residues: %s", longest_num_residues)
        logging.debug("Longest residue length: %s", longest_residue_length)

        bar.text("Padding residues")
        pdb_data = rpdb.pad_flatten(pdb_data, longest_num_residues, longest_residue_length)
        bar()

        bar.text('Exporting')
        f = open(pdb_data_file, 'wb')
        pickle.dump(pdb_data, f)
        f.close()
        bar()
        f = open(pdb_ids_file, 'wb')
        pickle.dump(pdb_ids, f)
        f.close()
        del pdbs, pdb_ids, pdb_data, f
        bar()

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    create(sdb)

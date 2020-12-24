import numpy as np
import pandas as pd
import pickle
import strings
import pdb

def create(pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
    print("--- PREPARING DATA ---")

    print("Reading PDB data...")
    pdbs = strings.get_pdbs()

    pdb_data = {}
    pdb_ids = {}
    for p in pdbs:
        pdb_data[p] = pdb.get_pdb_data(p, strings.pdbbind(p))
        pdb_ids[p] = strings.strings_id(p)

    f = open(pdb_data_file, 'wb')
    pickle.dump(pdb_data, f)
    f.close()
    f = open(pdb_ids_file, 'wb')
    pickle.dump(pdb_ids, f)
    f.close()

    print("Done exporting PDBs.")

import config
import strings
import numpy as np
import pandas as pd
import pickle
import pdb
import logging
from alive_progress import alive_bar

def create(sdb, pdb_data_file='pdb_data.pkl', pdb_ids_file='pdb_ids.pkl'):
    LIMIT = -1
    pdbs = sdb.get_pdbs()
    total = LIMIT if LIMIT != -1 else len(pdbs)
    total = 3*total+2

    with alive_bar(title='Parsing PDBs', total=total) as bar:
        bar.text('Building dictionaries')
        pdb_data = {}
        pdb_ids = {}
        longest_num_residues = -1
        longest_residue_length = -1
        for p in pdbs:
            d = pdb.get_pdb_data(p, sdb.pdbbind(p))
            num_residues = len(d)
            if num_residues > longest_num_residues:
                longest_num_residues = num_residues
            for residue in d:
                residue_length = len(residue)
                if residue_length > longest_residue_length:
                    longest_residue_length = residue_len

            pdb_data[p] = d
            pdb_ids[p] = sdb.strings_id(p)
            bar()
            if LIMIT != -1:
                LIMIT = LIMIT - 1
                if LIMIT == 0: break

        logging.debug("Longest number of residues: %s", longest_num_residues)
        logging.debug("Longest residue length: %s", longest_residue_length)

        pdb_data = pad_pdbs(bar, pdb_data, longest_num_residues, longest_residue_length)

        bar.text('Exporting')
        f = open(pdb_data_file, 'wb')
        pickle.dump(pdb_data, f)
        f.close()
        bar()
        f = open(pdb_ids_file, 'wb')
        pickle.dump(pdb_ids, f)
        f.close()
        bar()

def pad_pdbs(bar, pdb_data, longest_num_residues, longest_residue_length):
    bar.text("Padding residues")
    for pdb in pdb_data:
        pdb_data[pdb] = pad(pdb_data[pdb], longest_residue_length, [np.nan])
        bar()

    bar.text("Padding pdbs")
    pdb_data = pad_dict(pdb_data, longest_num_residues, [np.nan] * longest_residue_length)

    # bar.text("Flattening")
    # for pdb in pdb_data:
        # pdb_data[pdb] = flatten(pdb_data[pdb])
        # bar()

    return pdb_data

def flatten(arr):
    result = np.array([])
    for item in arr:
        if isinstance(item, list):
            result = np.concatenate([result, flatten(item)])
        else:
            result = np.append(result, [item])

    return result

def pad(to_pad, N, padder):
    for sublist in to_pad:
        sublist += padder * (N - len(sublist))

    return to_pad

def pad_dict(to_pad, N, padder):
    for k, v in to_pad:
        to_pad[k] = pad(v, N, padder)

    return to_padd

if __name__ == "__main__":
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    create(sdb)

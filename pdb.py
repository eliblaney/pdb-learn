import hashlib
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionWarning)
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser(PERMISSIVE=1)

    def get_pdb_data(structure_id, pdb_filename):
        """Get a representation of the PDB data of a particular protein"""
        s = parser.get_structure(structure_id, pdb_filename)
        rs = []
        for m in s:
            for c in m:
                for r in c:
                    # Encodes a residue ("ILE") as a number (23471766)
                    r_hashid = int(hashlib.sha1(r.get_resname().encode('UTF-8')).hexdigest(), 16) % (10 ** 8)
                    al = []
                    # The following adds each atom with its coordinates and b-factor
                    for a in r:
                        # Encodes the atom name as a number
                        a_hashid = int(hashlib.sha1(a.get_id().encode('UTF-8')).hexdigest(), 16) % (10 ** 8)
                        al.append(a_hashid)
                        al.append(a.get_bfactor())
                        al.append(a.get_coord())
                    # Add the array of residue hash + atom hash, b factor, coordinates for each atom and add
                    rs.append([r_hashid, al])

        # TODO: Add ligand binding, ligand primary sequences

        # Finally, flatten the array of residues
        return rs.ravel(order='C')

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
                    # The following adds each residue
                    hashid = int(hashlib.sha1(r.get_resname().encode('UTF-8')).hexdigest(), 16) % (10 ** 8)
                    rs.append(hashid)
                    # The following adds each atom with its coordinates and b-factor
#                    for a in r:
#                        hashid = int(hashlib.sha1(a.get_id().encode('UTF-8')).hexdigest(), 16) % (10 ** 8)
#                        rs.append(np.concatenate(([hashid, a.get_bfactor()], a.get_coord()), axis=None))
        return rs

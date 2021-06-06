import hashlib
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionWarning)
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser(PERMISSIVE=1)

    def get_pdb_data(structure_id, pdb_filename, pocket_filename):
        """Get a representation of the PDB data of a particular protein"""

        rs = build_residues(structure_id, pdb_filename, [])
        rs = build_residues(structure_id, pocket_filename, rs)

        # TODO: Add ligand primary sequences

        return rs
        # Finally, flatten the array of residues
        # return np.ravel(rs, order='C')


    def build_residues(structure_id, filename, rs):
        s = parser.get_structure(structure_id, filename)
        for m in s:
            for c in m:
                for r in c:
                    # Encodes a residue ("ILE") as a number (23471766)
                    r_hashid = int(
                        hashlib.sha1(
                            r.get_resname().encode('UTF-8')).hexdigest(),
                        16) % (10**8)
                    residue = [r_hashid]
                    # The following adds each atom with its coordinates and b-factor
                    for a in r:
                        # Encodes the atom name as a number
                        a_hashid = int(
                            hashlib.sha1(
                                a.get_id().encode('UTF-8')).hexdigest(),
                            16) % (10**8)
                        residue.append(a_hashid)
                        residue.append(a.get_bfactor())
                        np.concatenate([residue, a.get_coord()])
                    # Add the array of residue hash + atom hash, b factor, coordinates for each atom and add
                    rs.append(residue)

        return rs

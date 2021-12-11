import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PDBConstructionWarning)
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser(PERMISSIVE=1)

    def get_pdb_data(structure_id, pdb_filename):
        """Get a representation of the PDB data of a particular protein"""

        rs = build_residues(structure_id, pdb_filename, [])
        # rs = build_residues(structure_id, pocket_filename, rs)
        # ...can add more input data here

        return rs


    def build_residues(structure_id, filename, rs):
        s = parser.get_structure(structure_id, filename)
        residues = []
        atoms = []
        for m in s:
            for c in m:
                for r in c:
                    # Encodes a residue ("ILE") as a number
                    r_id = r.get_resname()
                    if r_id in residues:
                        r_id = residues.index(r_id)
                    else:
                        residues.append(r_id)
                        r_id = residues.index(r_id)
                    residue = [r_id]
                    # The following adds each atom with its coordinates and b-factor
                    for a in r:
                        # Encodes the atom name as a number
                        a_id = a.get_id()
                        if a_id in atoms:
                            a_id = atoms.index(a_id)
                        else:
                            atoms.append(a_id)
                            a_id = atoms.index(a_id)
                        residue.append(a_id)
                        residue.append(a.get_bfactor())
                        np.concatenate([residue, a.get_coord()])
                    # Add the array of residue hash + atom hash, b factor, coordinates for each atom and add
                    rs.append(residue)

        return rs

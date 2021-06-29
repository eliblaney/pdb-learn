import os
import pandas as pd
from alive_progress import alive_bar

class StringsDB:
    DIR_PDBBIND = None
    DIR_PDBBIND_GENERAL = None
    DIR_STRINGSDB = None

    pdbs = None
    pdbs_general = None

    DF_ALIASES = None
    DF_INFO = None
    DF_LINKS = None

    def __init__(self, pdbbind_refined='pdbbind-refined', pdbbind_general='pdbbind-general', stringsdbf='stringsdb-v11.0', version='11.0'):
        self.DIR_PDBBIND = pdbbind_refined
        self.DIR_PDBBIND_GENERAL = pdbbind_general
        self.DIR_STRINGSDB = stringsdbf

        with alive_bar(title='Importing databases') as bar:
            bar.text('pdbbind')
            self.pdbs = next(os.walk(self.DIR_PDBBIND))[1]
            self.pdbs_general = next(os.walk(self.DIR_PDBBIND_GENERAL))[1]
            bar()

            bar.text('stringsdb')
            self.DF_ALIASES = pd.read_csv(self.stringsdb('aliases', self.DIR_STRINGSDB, version), sep="\t")
            self.DF_INFO = pd.read_csv(self.stringsdb('info', self.DIR_STRINGSDB, version), sep="\t")
            self.DF_LINKS = pd.read_csv(self.stringsdb('links.full', self.DIR_STRINGSDB, version), sep="\s+")
            bar()

    def stringsdb(self, type, stringsdb_folder, version='11.0'):
        """Retrieve a filename pointing to the given STRING database type in the given folder"""
        species_id = 9606
        return os.path.join(stringsdb_folder, '{}.protein.{}.v{}.txt'.format(species_id, type, version))

    def pdbbind(self, pdb_id, general = False):
        """Retrieve a filename pointing to a particular protein specified by its PDB ID"""
        pdbbind = self.DIR_PDBBIND
        if general:
            pdbbind = self.DIR_PDBBIND_GENERAL
        return os.path.join(pdbbind, '{0}/{0}_protein.pdb'.format(pdb_id))

    def pdbbind_pocket(self, pdb_id, general = False):
        """Retrieve a filename pointing to a particular protein specified by its PDB ID"""
        pdbbind = self.DIR_PDBBIND
        if general:
            pdbbind = self.DIR_PDBBIND_GENERAL
        return os.path.join(pdbbind, '{0}/{0}_pocket.pdb'.format(pdb_id))

    def strings_id(self, pdb_id, aliases=None):
        """Get the STRING ID of a protein from its PDB ID"""
        if aliases is None:
            aliases = self.DF_ALIASES
        result = aliases.query('alias == "{}"'.format(pdb_id.upper()))
        return result.to_numpy()[0][0] if result.size > 0 else None

    def from_name(self, name, info=None):
        """Get the STRING ID of a protein from its name"""
        if info is None:
            info = self.DF_INFO
        result = info.query('preferred_name == "{}"'.format(name.upper()))
        return result.to_numpy()[0][0] if result.size > 0 else None

    def scores(self, p1, p2, links=None):
        """Retrieve all the scores for a particular protein-protein relationship"""
        if links is None:
            links = self.DF_LINKS
        result = links.query('protein1 == "{}" & protein2 == "{}"'.format(p1, p2))
        return result.to_numpy()[0][2:] if result.size > 0 else []

    def score(self, p1, p2):
        """Retrieve just the cumulative score for a particular protein-protein relationship"""
        result = self.scores(p1, p2)
        return result[13] if len(result) > 0 else -1

    def score_mapped(self, p1, p2):
        """Retrieve a mapped score for a particular protein-protein relationship"""
        return self.map_score(self.score(p1, p2))

    def get_pdbs(self, general = False):
        """Get all the pdbs in either the refined or general PDBbind sets"""
        if general:
            return self.pdbs_general
        return self.pdbs

    def map_score(self, score):
        """Map a score to a quantative identifier and a qualitative descriptor"""
        try:
            score = int(float(score))
        except:
            return (-1, 'Invalid')
        if score > 900:
            return (4, 'Highest')
        if score > 700:
            return (3, 'High')
        if score > 400:
            return (2, 'Medium')
        if score > 150:
            return (1, 'Low')
        return (0, 'None')

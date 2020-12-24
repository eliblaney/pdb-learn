import os
import pandas as pd

DIR_PDBBIND = None
DIR_PDBBIND_GENERAL = None
DIR_STRINGSDB = None

pdbs = None
pdbs_general = None

DF_ALIASES = None
DF_INFO = None
DF_LINKS = None

def stringsdb(type, stringsdb_folder, version='11.0'):
    """Retrieve a filename pointing to the given STRING database type in the given folder"""
    species_id = 9606
    return os.path.join(stringsdb_folder, '{}.protein.{}.v{}.txt'.format(species_id, type, version))

def pdbbind(pdb_id, general = False):
    """Retrieve a filename pointing to a particular protein specified by its PDB ID"""
    pdbbind = DIR_PDBBIND
    if general:
        pdbbind = DIR_PDBBIND_GENERAL
    return os.path.join(pdbbind, '{0}/{0}_protein.pdb'.format(pdb_id))

def setup(pdbbind_refined='pdbbind-refined', pdbbind_general='pdbbind_general', stringsdb='stringsdb-v11.0', version='11.0'):
    """Load the STRING and PDBbind databases"""
    DIR_PDBBIND = pdbbind_refined
    DIR_PDBBIND_GENERAL = pdbbind_general
    DIR_STRINGSDB = stringsdb

    pdbs = next(os.walk(DIR_PDBBIND))[1]
    pdbs_general = next(os.walk(DIR_PDBBIND_GENERAL))[1]

    if DF_ALIASES is None:
        DF_ALIASES = pd.read_csv(stringsdb('aliases', DIR_STRINGSDB, version), sep="\t")
    if DF_INFO is None:
        DF_INFO = pd.read_csv(stringsdb('info', DIR_STRINGSDB, version), sep="\t")
    if DF_LINKS is None:
        DF_LINKS = pd.read_csv(stringsdb('links.full', DIR_STRINGSDB, version), sep="\s+")

def strings_id(pdb_id, aliases=DF_ALIASES):
    """Get the STRING ID of a protein from its PDB ID"""
    result = aliases.query('alias == "{}"'.format(pdb_id.upper()))
    return result.to_numpy()[0][0] if result.size > 0 else None

def from_name(name, info=DF_INFO):
    """Get the STRING ID of a protein from its name"""
    result = info.query('preferred_name == "{}"'.format(name.upper()))
    return result.to_numpy()[0][0] if result.size > 0 else None

def scores(p1, p2, links=DF_LINKS):
    """Retrieve all the scores for a particular protein-protein relationship"""
    result = links.query('protein1 == "{}" & protein2 == "{}"'.format(p1, p2))
    return result.to_numpy()[0][2:] if result.size > 0 else []

def score(p1, p2):
    """Retrieve just the cumulative score for a particular protein-protein relationship"""
    result = scores(p1, p2)
    return result[13] if len(result) > 0 else -1

def score_mapped(p1, p2):
    """Retrieve a mapped score for a particular protein-protein relationship"""
    return map_score(score(p1, p2))

def get_pdbs(general = False):
    """Get all the pdbs in either the refined or general PDBbind sets"""
    if general:
        return pdbs_general
    return pdbs

def map_score(score):
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

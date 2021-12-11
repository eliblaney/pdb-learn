import logging
import os

def exists(files):
    """Determines if a file or list of files all exist"""
    if type(files) == list:
        found = []
        notfound = []
        for f in files:
            if exists(f):
                found.append(f)
            else:
                notfound.append(f)
        if len(notfound) > 0:
            logging.warning("The following files were found: %s, but some were missing: %s", ", ".join(found), ", ".join(notfound))
            return False
        return True

    return os.path.exists(files)
            

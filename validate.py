import logging
import os

def exists(file):
    """Determines if a file or list of files all exist"""
    if type(file) == list:
        # Validate each file or folder
        found = []
        notfound = []
        for f in file:
            if exists(f):
                found.append(f)
            else:
                notfound.append(f)
        if len(found) > 0 and len(notfound) > 0:
            logging.warning("The following files were found: %s, but some were missing: %s", ", ".join(found), ", ".join(notfound))
            return False
        return True
    elif os.path.isdir(file):
        # Validate if there is anything inside the folder
        return any(os.scandir(file))

    # Validate if a single file exists
    return os.path.exists(file)
            

import os
import sys
import logging
import datetime
import createpdbs
import train
import strings
import predict
from alive_progress import config_handler

def main():
    LOG_DIR = 'logs/'

    now = datetime.datetime.now()
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logging.basicConfig(filename="{}/{}.log".format(LOG_DIR, now), encoding='utf-8', level=logging.DEBUG, filemode='w', format='[%(asctime)s %(levelname)8s] %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.debug("Program started %s", now)

    config_handler.set_global(unknown='dots_waves', bar='blocks')
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    logging.info("Exporting PDBs...")
    # Read databases and store PDB representations
    createpdbs.create(sdb)
    logging.info("Finished exporting PDBs.")

    logging.info("Starting training")
    # Train all models
    max_length = train.train(sdb)

    logging.info("Starting predictions")
    # Run model predictions
    predict.predict(sdb, num=100, each=100, max_length=max_length)

if __name__ == "__main__":
    main()

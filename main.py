import config
import logging
import strings
import createpdbs
import train
import predict
import builder
import validate

def main():
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    if not validate.exists(['pdb_ids.pkl', 'pdb_data.pkl']):
        logging.info("Exporting PDBs...")
        # Read databases and store PDB representations
        createpdbs.create(sdb)
        logging.info("Finished exporting PDBs.")
    else:
        logging.info("Found exported PDBs, skipping creation phase..")

    if not validate.exists('data/'):
        logging.info("Building inputs and outputs...")
        builder.PDBBuilder(sdb).partition()
    else:
        logging.info("Found data folder, skipping partioning phase.")

    if not validate.exists('saved_models/'):
        logging.info("Starting training")
        # Train all models
        max_length = train.train(sdb)

        if validate.exists('saved_models/'):
            logging.info("Starting predictions")
            # Run model predictions
            predict.predict(sdb, num=100, each=100, max_length=max_length)
        else:
            logging.warning("Could not find saved models after training, skipping predictions phase.")
    else:
        logging.info("Found saved models, skipping training and predicting phases.")

if __name__ == "__main__":
    main()

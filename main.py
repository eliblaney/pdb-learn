import config
import logging
import strings
import createpdbs
import train
import predict
import builder

def main():
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    logging.info("Exporting PDBs...")
    # Read databases and store PDB representations
    createpdbs.create(sdb)
    logging.info("Finished exporting PDBs.")

    logging.info("Building inputs and outputs...")
    builder.build(sdb)

    logging.info("Starting training")
    # Train all models
    max_length = train.train(sdb)

    logging.info("Starting predictions")
    # Run model predictions
    predict.predict(sdb, num=100, each=100, max_length=max_length)

if __name__ == "__main__":
    main()

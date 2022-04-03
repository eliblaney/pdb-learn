import os
import config
import logging
import strings
import validate

def main():
    logging.info("Importing databases...")
    sdb = strings.StringsDB()
    logging.info("Finished importing databases.")

    if not validate.exists('pdbs/'):
        if not os.exists('pdbs/'):
            os.mkdir('pdbs/')
        import createpdbs
        logging.info("Exporting PDBs...")
        # Read databases and store PDB representations
        limit=1500
        for i in range(3):
            logging.debug("Creating PDB files using createpdbs.create({}, {})...".format(limit, i*limit))
            createpdbs.create(sdb, pdb_data_file='pdbs/pdb_data_' + str(i) + '.pkl', pdb_ids_file='pdbs/pdb_ids_' + str(i) + '.pkl', limit=limit, offset=i*limit)
        logging.info("Finished exporting PDBs.")
    else:
        logging.info("Found exported PDBs, skipping creation phase..")

    if not validate.exists('data/'):
        import builder
        logging.info("Building inputs and outputs...")
        builder.PDBBuilder(sdb).partition(cpus=1)
    else:
        logging.info("Found data folder, skipping partioning phase.")

    if not validate.exists('saved_models/'):
        import train
        logging.info("Starting training")
        # Train all models
        max_length = train.train_partitioned(sdb)

        if validate.exists('saved_models/'):
            import predict
            logging.info("Starting predictions")
            # Run model predictions
            predict.predict(sdb, num=100, each=100, max_length=max_length)
        else:
            logging.warning("Could not find saved models after training, skipping predictions phase.")
    else:
        logging.info("Found saved models, skipping training and predicting phases.")

if __name__ == "__main__":
    main()

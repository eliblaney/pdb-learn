import sys
import logging
import datetime
import createpdbs
import train

def main():
    now = datetime.datetime.now()
    logging.basicConfig(filename="{}.log".format(now), encoding='utf-8', level=logging.DEBUG, filemode='w', format='[%(asctime)s %(levelname)8s] %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.debug("Program started %s", now)

    # Read databases and store PDB representations
    createpdbs.create()

    # Train all models
    (models, max_length) = train.train()

    # Run model predictions
    import predict
    predict.predict(num=100, each=100, max_length=max_length, models=models)

if __name__ == "__main__":
    main()

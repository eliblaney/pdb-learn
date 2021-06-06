import createpdbs
import train

def main():
    createpdbs.create()

    (models, max_length) = train.train()

    import predict
    predict.predict(num=100, each=100, max_length=max_length, models=models)

if __name__ == "__main__":
    main()

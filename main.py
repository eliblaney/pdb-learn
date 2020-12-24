import createpdbs
import train
import predict

def main():
    createpdbs.create()
    (models, max_length) = train.train()
    predict.predict(num=100, each=100, max_length=max_length, models=models)

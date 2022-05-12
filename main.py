from data import data_preprocessing
from train import train_model


def main():
    all_data = data_preprocessing(amount_of_rows=40)
    f1 = train_model(all_data, verbose=True, t=0.5)


# main()

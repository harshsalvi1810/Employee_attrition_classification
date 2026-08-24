from src.data_ingestion import data_loader
from src.data_preprocessing import preprocessing
from src.model_build import model_build


def main():
    df = data_loader()
    print(df.shape)

    X_train,X_test,y_train,y_test = preprocessing(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    model = model_build(X_train,X_test,y_train,y_test)


main()
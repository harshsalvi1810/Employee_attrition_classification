import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def preprocessing(df):

    df.drop_duplicates()

    df.drop(columns = 'Employee ID')

    X = df.drop(columns = ['Attrition'])
    y = df['Attrition']

    numerical_cols = X.select_dtypes(exclude="object").columns
    categorical_cols = X.select_dtypes(include="object").columns

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False
        ))
    ])

    transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    sm=SMOTE()
    X_train,y_train=sm.fit_resample(X_train,y_train)

    pca = PCA(n_components=0.95, random_state=1)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test
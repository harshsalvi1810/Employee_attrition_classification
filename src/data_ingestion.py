import numpy as numpy
import pandas as pd

def data_loader():

    df = pd.read_csv(r'C:\Employee_attrition_classification\data\test.csv')

    return df
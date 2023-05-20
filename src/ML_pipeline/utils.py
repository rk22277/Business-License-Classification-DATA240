import pandas as pd
from sklearn.model_selection import train_test_split

#Function to read the data
def read_data(file_path, **kwargs):
    raw_data=pd.read_csv(file_path  ,**kwargs)
    return raw_data

# Function to split data into training and testing
def data_split(train_data,test_data, size=0.1):
    return train_test_split(train_data,test_data, test_size=size)
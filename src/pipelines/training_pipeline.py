import os,sys
import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    print(train_path,test_path)
    data_transform=DataTransformation()
    train_array,test_array=data_transform.initiate_data_transform(train_path=train_path,test_path=test_path)
    
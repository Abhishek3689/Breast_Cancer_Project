import os,sys
import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    print(train_path,test_path)
    data_transform=DataTransformation()
    train_array,test_array=data_transform.initiate_data_transform(train_path=train_path,test_path=test_path)
    model_object=ModelTrainer()
    best_model,best_score=model_object.initiate_model_training(train_array,test_array)
    print(f"Best Model Found , Model Name : {best_model} , accuracy Score : {best_score}")
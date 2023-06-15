import os,sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score


def save_object(filepath,obj):
    try:
        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(filepath):
    with open(filepath,'rb') as file_obj:
        pickle.load(file_obj)

def Evaluate_model(models,X_train,X_test,y_train,y_test):
    report={}
    for i in range(len(models)):
        model=list(models.values())[i]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        score=accuracy_score(y_test,y_pred)
        report[list(models.keys())[i]]=score
    return report
import os,sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException

def save_object(filepath,obj):
    try:
        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(filepath):
    with open(filepath,'rb') as file_obj:
        pickle.load(file_obj)
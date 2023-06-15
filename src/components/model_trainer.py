import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object,save_object,Evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    model_trained_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Model training is initiated")
            os.makedirs(os.path.dirname(self.model_trainer_config.model_trained_path),exist_ok=True)
            X_train,X_test,y_train,y_test=train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1]

            logging.info("Data preprocessed data has been splitted")
            models={'Naive Bayes':GaussianNB(),
                    'Logistic Regression':LogisticRegression(),
                    'Decision Tree':DecisionTreeClassifier(),
                    'Random Forest':RandomForestClassifier(),
                    'SVM':SVC(),
                    'KNN':KNeighborsClassifier()}
            
            reports=Evaluate_model(models,X_train,X_test,y_train,y_test)
            logging.info(f"Reports has been completed and results are [{reports}]")
            best_score=max((list(reports.values())))
            best_model_name=list(reports.keys())[np.argmax(list(reports.values()))]
            best_model=models[best_model_name]
            logging.info(f"Best model is [{best_model_name}] and best score is [{best_score}]")
            print(f"Best model is [{best_model_name}] and best score is [{best_score}]")
            save_object(
                self.model_trainer_config.model_trained_path,best_model
            )
            logging.info("Model has been saved")
            return(best_model_name,best_score)
        except Exception as e:
            logging.info("Error Occurred in Model traaining")
            raise CustomException(e,sys)

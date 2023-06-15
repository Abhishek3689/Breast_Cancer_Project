import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_preprocessor_file(self):
        try:
            logging.info("Pipeline preprocessor is initiated")
            num_col=['mean texture',
                    'mean smoothness',
                    'mean compactness',
                    'mean concave points',
                    'mean symmetry',
                    'mean fractal dimension',
                    'texture error',
                    'area error',
                    'smoothness error',
                    'compactness error',
                    'concavity error',
                    'concave points error',
                    'symmetry error',
                    'fractal dimension error',
                    'worst texture',
                    'worst area',
                    'worst smoothness',
                    'worst compactness',
                    'worst concavity',
                    'worst concave points',
                    'worst symmetry',
                    'worst fractal dimension']
            num_pipeline=Pipeline(steps=
                                [('Imputer',SimpleImputer(strategy='median')),
                                    ('Scaler',StandardScaler())])
            preprocessor=ColumnTransformer([('Numerical Pipeline',num_pipeline,num_col)])
            logging.info("prprocessor pipeline is completed")
            return preprocessor
        except Exception as e:
            logging.info("Error occured while preprocessor pipeline")
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            logging.info("Preprocessor object creation is initiated")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            train_df['target']=train_df['target'].map({'malignant':1,'benign':0})
            test_df['target']=test_df['target'].map({'malignant':1,'benign':0})
            target_col='target'
            drop_col=['mean radius',
                    'mean perimeter',
                    'mean area',
                    'mean concavity',
                    'radius error',
                    'perimeter error',
                    'worst radius',
                    'worst perimeter',target_col]
            
            train_feature_df=train_df.drop(drop_col,axis=1)
            train_target_df=train_df[target_col]

            test_feature_df=test_df.drop(drop_col,axis=1)
            test_target_df=test_df[target_col]
            logging.info("cleaned trained and test data is obtained")
            preprocessor=self.get_preprocessor_file()
            train_feature_scaled=preprocessor.fit_transform(train_feature_df)
            test_feature_scaled=preprocessor.transform(test_feature_df)

            train_arr=np.c_[train_feature_scaled,np.array(train_target_df)]
            test_arr=np.c_[test_feature_scaled,np.array(test_target_df)]
            logging.info("preproessor object and array is created ")
            save_object(self.data_transformation_config.preprocessor_path,preprocessor)
            logging.info("preprocessor object is saved ")
            return(
                train_arr,test_arr
            )
        except Exception as e:
            logging.info("Error occured in getting preprocessor object")
            raise CustomException(e,sys)


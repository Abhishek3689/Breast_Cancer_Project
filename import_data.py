import pandas as pd
import numpy as np
import os,sys
import certifi

from src.logger import logging
from src.exception import CustomException
import json
import pymongo

ca = certifi.where()
# Making connection with pymongo
try:
    logging.info("Uploading Data is initiated")
    from pymongo.mongo_client import MongoClient
    uri = "mongodb+srv://abhisheknishad:abhisheknishad@cluster0.rgawbxa.mongodb.net/?retryWrites=true&w=majority"
    client=MongoClient(uri, tlsCAFile=ca)
    logging.info("Connection with mongodb is established")
    from sklearn.datasets import load_breast_cancer

    ## Load Dataset from sklearn dataset libray
    breast_cancer=load_breast_cancer()
    X=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
    y=pd.DataFrame(breast_cancer.target,columns=['target'])

    ## mapping target variable with thei respective names
    y['target']=y['target'].map({0:'benign',1:'malignant'})

    ## Combine the dataframes 
    df=pd.concat([X,y],axis=1)
    logging.info("DataFrame has been created")
    ## Creating Database and collection in Mongodb
    db_name='Breast_Cancer'
    collection_name='cancer_data'
    collection=client[db_name][collection_name]

    ## Creating json format and saving  it into Database
    json_report=list(json.loads(df.T.to_json()).values())
    collection.insert_many(json_report)
    logging.info("Data has been uploaded in Mongodb")
except Exception as e:
    logging.info("Error occured while uploading data in Mongo DB")
    raise CustomException(e,sys)
    
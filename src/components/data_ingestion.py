import os,sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pymongo

# Making connection with pymongo
from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://abhisheknishad:abhisheknishad@cluster0.rgawbxa.mongodb.net/?retryWrites=true&w=majority"
client=MongoClient(uri)
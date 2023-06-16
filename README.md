## Breat Cancer Detection
### The objective of the Project is to detect Breast Cancer as Malignant or Benign with sets of given clinical parameter
#### Steps involves creating environment and project structure using Automated Pipeline Structure
- Data Ingestion: This involves Extracting raw Data from Mongo DB to the system
- Data Transformation: In this step we perform Data Cleaning, Preprocssing,Feature Engineering ,feature selection and to save preprocessor pickle file
- Model Trainer: This step involves training and evaluation using multiple ML models, Selecting Best Model based on best metrics,accuracy_score and saving best model in pickle.
- Prediction : This steps involves lodaing of preprocessor and model file for prediction of unseen data
- Web App and Prediction: Creation of Web app for front end users to input clinical parameter and predict the results as 'Malignant' or 'Benign"
- Deployement: Deploy in AWS using Elastic beanstalk

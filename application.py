from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import  Predict_data,Customised_Data
from src.logger import logging
from src.exception import CustomException
import os,sys

application=Flask(__name__)
app=application

@app.route('/')
def HomePage():
    return render_template('index.html')

@app.route('/predict_disease',methods=['POST','GET'])
def predict_cancer():
    try:
         
        if request.method=='GET':
            return render_template('home.html')
    
        else:
            logging.info("form input is initiated ")
            data=Customised_Data(
                mean_texture=float(request.form['mean texture']),
                mean_smoothness=float(request.form['mean smoothness']),
                mean_compactness=float(request.form['mean compactness']),
                mean_concave_points=float(request.form['mean concave points']),
                mean_symmetry=float(request.form['mean symmetry']),
                mean_fractal_dimension=float(request.form['mean fractal dimension']),
                texture_error=float(request.form['texture error']),
                area_error=float(request.form['area error']),
                smoothness_error=float(request.form['smoothness error']),
                compactness_error=float(request.form['compactness error']),
                concavity_error=float(request.form['concavity error']),
                concave_points_error=float(request.form['concave points error']),
                symmetry_error=float(request.form['symmetry error']),
                fractal_dimension_error=float(request.form['fractal dimension error']),
                worst_texture=float(request.form['worst texture']),
                worst_area=float(request.form['worst area']),
                worst_smoothness=float(request.form['worst smoothness']),
                worst_compactness=float(request.form['worst compactness']),
                worst_concavity=float(request.form['worst concavity']),
                worst_concave_points=float(request.form['worst concave points']),
                worst_symmetry=float(request.form['worst symmetry']),
                worst_fractal_dimension=float(request.form['worst fractal dimension'])
                
            )
            logging.info("form input completed")
            df=data.get_dataframe()
            logging.info(f"{df.head()}")
            logging.info("test data is converted into dataframe")
            
            output=Predict_data.predict(df)
            logging.info("Result is obtained")
            if output==1:
                results='malignant'
            else:
                results='benign'
            return render_template('home.html',results=results)
        
    except Exception as e:
        logging.info("error occured in processing app through form")
        raise CustomException(e,sys)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)



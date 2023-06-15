import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.utils import load_object

class Customised_Data:
    def __init__(self,mean_texture:float,
                    mean_smoothness:float,
                    mean_compactness:float,
                    mean_concave_points:float,
                    mean_symmetry:float,
                    mean_fractal_dimension:float,
                    texture_error:float,
                    area_error:float,
                    smoothness_error:float,
                    compactness_error:float,
                    concavity_error:float,
                    concave_points_error:float,
                    symmetry_error:float,
                    fractal_dimension_error:float,
                    worst_texture:float,
                    worst_area:float,
                    worst_smoothness:float,
                    worst_compactness:float,
                    worst_concavity:float,
                    worst_concave_points:float,
                    worst_symmetry:float,
                    worst_fractal_dimension:float):
        self.mean_texture=mean_texture
        self.mean_smoothness=mean_smoothness
        self.mean_compactness=mean_compactness
        self.mean_concave_points=mean_concave_points
        self.mean_symmetry=mean_symmetry
        self.mean_fractal_dimension=mean_fractal_dimension
        self.texture_error=texture_error
        self.area_error=area_error
        self.smoothness_error=smoothness_error
        self.compactness_error=compactness_error
        self.concavity_error=concavity_error
        self.concave_points_error=concave_points_error
        self.symmetry_error=symmetry_error
        self.fractal_dimension_error=fractal_dimension_error
        self.worst_texture=worst_texture
        self.worst_area=worst_area
        self.worst_smoothness=worst_smoothness
        self.worst_compactness=worst_compactness
        self.worst_concavity=worst_concavity
        self.worst_concave_points=worst_concave_points
        self.worst_symmetry=worst_symmetry
        self.worst_fractal_dimension=worst_fractal_dimension

    def get_dataframe(self):
        custom_data={'mean texture':[self.mean_texture],
                     'mean smoothness':[self.mean_smoothness],
                     'mean compactness':[self.mean_compactness],
                     'mean concave points':[self.mean_concave_points],
                     'mean symmetry':[self.mean_symmetry],
                     'mean fractal dimension':[self.mean_fractal_dimension],
                     'texture error':[self.texture_error],
                     'area error':[self.area_error],
                     'smoothness error':[self.smoothness_error],
                     'compactness error':[self.compactness_error],
                     'concavity error':[self.concavity_error],
                     'concave points error':[self.concave_points_error],
                     'symmetry error':[self.symmetry_error],
                     'fractal dimension error':[self.fractal_dimension_error],
                     'worst texture':[self.worst_texture],
                     'worst area':[self.worst_area],
                     'worst smoothness':[self.worst_smoothness],
                     'worst compactness':[self.worst_compactness],
                     'worst concavity':[self.worst_concavity],
                     'worst concave points':[self.worst_concave_points],
                     'worst symmetry':[self.worst_symmetry],
                     'worst fractal dimension':[self.worst_fractal_dimension]}
        df=pd.DataFrame(custom_data)
        print(df)
        logging.info("Dataframe has been gathered")
        return df
    
class Predict_data:
    def __init__(self):
        pass

    def predict(features):
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        model_path=os.path.join('artifacts','model.pkl')
       
        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)
        logging.info("Model and preprocessor has been loaded")
        
        X_scaled=preprocessor.transform(features)
        logging.info("Data has been transformed")
        y_pred=model.predict(X_scaled) 
        logging.info(f"The Result of input paramters is [{y_pred}]")     
        return y_pred


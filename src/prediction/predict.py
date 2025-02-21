from src.logging import logger
from src.exception.exception import ProjectException
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import pickle
import pandas as pd 
import numpy as np
import sys, os

@dataclass
class ModelConfig:
    def __init__(self):
        self.models: List[str] = ['adaboost', 'xgb', 'lgbm', 'xgbrf']
        self.oob_model: str = 'oob_model'
        # self.root_path: str = 'output/models'
        self.root_path: str = os.path.join(os.getcwd(),'output/models')

class Predictor:
    def __init__(self, models: List[str] = None)->None:
        """
        Initiallization of Preditor
        parameter:
            models: List[str] = list of names of models
        """
        self.config = ModelConfig()
        self.root_path = self.config.root_path
        self.models = self.config.models if models is None else models
        self._oob_model_name = self.config.oob_model
        self.loader()
    @staticmethod
    def convert(X:Dict)->pd.DataFrame:
        df = X.copy()
        columns = ['Brand','Material', 'Size', 'Compartments' , 'Laptop Compartment' ,
            'Waterproof', 'Style' , 'Color' , 'Weight Capacity (kg)' ,]
        for feature in columns:
            df[feature] = [X[feature],]
        return pd.DataFrame(df)
    
    def impute(self)->None:
        """
        Impute the feaures.
        return: None
        """
        imputed_vals = {
            'Brand' : 'Adidas',
            'Material' : 'Polyester',
            'Size' : 'Medium',
            'Compartments' : 1,
            'Laptop Compartment' : 'Yes',
            'Waterproof' : 'Yes',
            'Style' : 'Messenger',
            'Color' : 'Pink',
            'Weight Capacity (kg)' : 18
        }
        for feature in imputed_vals.keys():
            if self.X[feature] == 'None':
                self.X[feature] = [imputed_vals[feature]]
            else:
                self.X[feature] = [self.X[feature]]
        
        self.X = pd.DataFrame(self.X)

    def extract_features(self)->None:
        """
        Extracts new features.
        return: None
        """
        self.X['Compartments'] = self.X['Compartments'].astype(int).astype(str)
        self.X['Brand_Material'] = self.X['Brand'] + '_' + self.X['Material']
        self.X['Brand_Size'] = self.X['Brand'] + '_' + self.X['Size']
        self.X['Brand_Waterproof'] = self.X['Brand'] + '_' + self.X['Waterproof']
        self.X['Brand_Style'] = self.X['Brand'] + '_' + self.X['Style']
        self.X['Brand_Color'] = self.X['Brand'] + '_' + self.X['Color']
        self.X['Material_Size'] = self.X['Material'] + '_' + self.X['Size']
        self.X['Material_Compartments'] = self.X['Material'] + '_' + self.X['Compartments']
        self.X['Material_Laptop'] = self.X['Material'] + '_' + self.X['Laptop Compartment']
        self.X['Material_Waterproof'] = self.X['Material'] + '_' + self.X['Waterproof']
        self.X['Material_Style'] = self.X['Material'] + '_' + self.X['Style']
        self.X['Material_Color'] = self.X['Material'] + '_' + self.X['Color']
        self.X['Size_Style'] = self.X['Size'] + '_' + self.X['Style']
        self.X['Size_Color'] = self.X['Size'] + '_' + self.X['Color']
        self.X['Laptop_Style'] = self.X['Laptop Compartment'] + '_' + self.X['Style']
        self.X['Laptop_Color'] = self.X['Laptop Compartment'] + '_' + self.X['Color']
        self.X['Waterproof_Color'] = self.X['Waterproof'] + '_' + self.X['Color']
        self.X['Style_Color'] = self.X['Style'] + '_' + self.X['Color']
        self.X['Brand_Material_Style'] = self.X['Brand'] + '_' + self.X['Material'] + '_' + self.X['Style']
        self.X['Brand_Size_Laptop'] = self.X['Brand'] + '_' + self.X['Size'] + '_' + self.X['Laptop Compartment']
        self.X['Brand_Size_Style'] = self.X['Brand'] + '_' + self.X['Size'] + '_' + self.X['Style']
        self.X['Brand_Size_Color'] = self.X['Brand'] + '_' + self.X['Size'] + '_' + self.X['Color']
        self.X['Brand_Waterproof_Color'] = self.X['Brand'] + '_' + self.X['Waterproof'] + '_' + self.X['Color']
        self.X['Brand_Style_Color'] = self.X['Brand'] + '_' + self.X['Style'] + '_' + self.X['Color']
        self.X['Material_Size_Style'] = self.X['Material'] + '_' + self.X['Size'] + '_' + self.X['Style']
        self.X['Material_Laptop_Color'] = self.X['Material'] + '_' + self.X['Laptop Compartment'] + '_' + self.X['Color']
        self.X['Material_Style_Color'] = self.X['Material'] + '_' + self.X['Style'] + '_' + self.X['Color']
        
    
    def loader(self)->None:
        """
        Loads important files.
        return : None
        """
        self.scaler, self.encoder = None, None
        self._trained_models = []
        try:
            with open(f'{self.root_path}/scaler.pkl','rb') as f:
                self.scaler = pickle.load(f)
                logger.logging.info("scaler.pkl is successfully read!")
        except:
            logger.logging.info("scaler.pkl is not found in output/models.")

        try:
            with open(f'{self.root_path}/encoder.pkl','rb') as f:
                self.encoder = pickle.load(f)
                logger.logging.info("encoder.pkl is successfully read!")
        except:
            logger.logging.info("encoder.pkl is not found in output/models.")
        
        for model_name in self.models:
            try:
                with open(f'{self.root_path}/{model_name}.pkl','rb') as f:
                    model = pickle.load(f)
                self._trained_models.append(model)
                logger.logging.info(f"{model_name}.pkl is successfully read!")
            except Exception as e:
                logger.logging.info(f"{model_name}.pkl is not found in output/models. Error: {e}")
                print(f"Model {model_name} failed to load due to {e}")
        
        try:
            with open(f'{self.root_path}/{self._oob_model_name}.pkl','rb') as f:
                self._oob_model = pickle.load(f)
                logger.logging.info(f"{self._oob_model_name}.pkl is successfully read!")
        except:
            logger.logging.info(f"{self._oob_model_name}.pkl is not found in output/models.")
    
    def transform(self)-> pd.DataFrame | None:
        """
        Transforms numerical and categorical features respectively.
        return: pd.DataFrame
        """
        # Check whether attributes num_features anad cat_features exists
        if not (hasattr(self,'num_features') or hasattr(self,'cat_features')):
            logger.logging.info("There is no numerical and categorical features!")
            return None
        # keeping a copy of data
        X_new = self.X.copy()
        if self.scaler is not None:
            try: 
                X_new[self.num_features] = self.scaler.transform(self.X[self.num_features])
            except ValueError:
                logger.logging.info("Transformation failed! Raised ValueError!")
            except Exception as e:
                raise ProjectException(e,sys)
        else:
            logger.logging.info("Scaler is set as None. We are unable to convert it.")

        if self.encoder is not None:
            try: 
                X_new[self.cat_features] = self.encoder.transform(self.X[self.cat_features])
            except ValueError:
                logger.logging.info("Transformation failed! Raised ValueError!")
            except Exception as e:
                raise ProjectException(e,sys)
        else:
            logger.logging.info("Encoder is set as None. We are unable to convert it.")
        return X_new
            

    def _predict_models(self)-> None:
        """
        Predicts the different models
        return: None
        """
        self._predicted_values = {}
        for model_name, model in zip(self.models, self._trained_models):
            try:
                self._predicted_values[model_name] = model.predict(self._X)
            except Exception as e:
                raise ProjectException(e,sys)
        
        self._oob_pred = None
        try:
            self._oob_pred = self._oob_prediction()
        except Exception as e:
            raise ProjectException(e,sys)

    def _oob_prediction(self)->float| Dict |  None:
        """
        Out-of-box prediction
        return: float or None
        """    
        try:
            return self._oob_model.predict(pd.DataFrame(self._predicted_values))
        except Exception as e:
            logger.logging.info(f"We failed to predict due to {e}!")
            return None
        
    def predict(self, X: Dict = None)->Dict:
        """
        Predicts the estimation.
        parameters:
            X : dict
        return: dict
        """

        self.X = X
        self.impute()
        self.extract_features()
        self.loader()
        self.num_features = ['Weight Capacity (kg)']
        self.cat_features = self.X.select_dtypes(include=['object']).columns
        self._X = self.transform()
        
        if self._X is None:
            logger.logging.info("We can not predict. Investigate into Categorical and Numerical Features!")
            return {}
        
        self._predict_models()
        result = self._predicted_values.copy()
        for model_name in self.models:
            result[model_name] = str(result[model_name][0].round(2))
        result['final prediction'] = str(self._oob_pred[0].round(2))

        return result



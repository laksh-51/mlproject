import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        
        try:
            numerical_column = ["writing_score","reading_score"]
            categorial_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                
            ]
            
            num_pipeline= Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                
                ]
            )    
                
            cat_pipeline=Pipeline(
                
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    
                    
                ]
                
                
            )
            
            logging.info(f"Categorial columns: {categorial_columns}" )
            logging.info(f"Numerial columns: {numerical_column}" )
                        
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_column),
                    ("cat_pipelines", cat_pipeline, categorial_columns)
                    
                ]
            )
            
            
            return preprocessor
        
        
        except Exception as e:
            raise CustomException
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Tead train and test data completed")
            
            logging.info("Obtaining preprocessing object ")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_column = ["writing_score","reading_score"]
            
            input_freature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_freature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on traing dataframe and testing dataframe."
            )
            
            input_freature_train_arr=preprocessing_obj.fit_transform(input_freature_train_df)
            input_freature_test_arr=preprocessing_obj.transform(input_freature_test_df)
            
            train_arr = np.c_[
                input_freature_train_arr,np.array(target_feature_train_df)
                
            ]
            test_arr = np.c_[
                input_freature_test_arr,np.array(target_feature_test_df)
                
            ]
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
            
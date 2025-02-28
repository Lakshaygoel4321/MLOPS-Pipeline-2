from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object
import sys
import os
import pandas as pd
import numpy as np

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation
        """
        try:
            
            numerical_column = ['Pclass','Age','Fare','Family Persion']
            categorical_column = [
                    'Sex',
                    'Embarked'
                    ]

            num_column = Pipeline(steps=[
                ('numerical_col',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            
            ])

            cat_column = Pipeline(steps=[
                ('categorical_column',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'categorical column: {categorical_column}')
            logging.info(f"numerical column: {numerical_column}")

            preprocessor = ColumnTransformer([
                ('numerical_preprocess',num_column,numerical_column),
                ('categorical_preprocessor',cat_column,categorical_column)
             ])
            
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        

    
    def initiate_data_transormation(self,train_path,test_path):

        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name="Survived"
            numerical_columns = ['Pclass','Age','Fare','Family Persion']
            # train
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_train_df = train_df[target_column_name]
            # test
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformer_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

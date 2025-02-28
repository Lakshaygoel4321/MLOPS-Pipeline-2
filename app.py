from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_transformation import DataTransformer
from src.mlproject.components.data_transformation import DataTransformerConfig
import os
import sys

if __name__ == "__main__":
    logging.info("The execution is started")



try:

    data_ingestion = DataIngestion()
    train_path_file,test_path_file = data_ingestion.initiate_data_ingestion()

    data_transformer = DataTransformer()
    data_transformer.initiate_data_transormation(train_path_file,test_path_file)

    
except Exception as e:
    raise CustomException(e,sys)


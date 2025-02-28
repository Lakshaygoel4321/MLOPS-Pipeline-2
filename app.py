from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_transformation import DataTransformer
from src.mlproject.components.data_transformation import DataTransformerConfig
from src.mlproject.components.model_tranier import ModelTrainer,ModelTrainerConfig
import os
import sys

if __name__ == "__main__":
    logging.info("The execution is started")



try:

    data_ingestion = DataIngestion()
    train_path_file,test_path_file = data_ingestion.initiate_data_ingestion()

    data_transformer = DataTransformer()
    train_path,test_path,_ = data_transformer.initiate_data_transormation(train_path_file,test_path_file)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_path,test_path))

except Exception as e:
    raise CustomException(e,sys)


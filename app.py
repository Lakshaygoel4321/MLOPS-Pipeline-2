from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import os
import sys

if __name__ == "__main__":
    logging.info("The execution is started")



try:

    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()


except Exception as e:
    raise CustomException(e,sys)


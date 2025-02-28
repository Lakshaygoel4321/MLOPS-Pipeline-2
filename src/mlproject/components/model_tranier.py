import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models
import dagshub
dagshub.init(repo_owner='iamhimanshu12goel', repo_name='MLOPS-Pipeline-2', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/iamhimanshu12goel/MLOPS-Pipeline-2.mlflow")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        f1 = f1_score(actual,pred)
        recall = recall_score(actual,pred)
        precision = precision_score(actual,pred)
        return f1,recall,precision

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                "Decision Tree": {
                    'criterion':['gini','entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.01,.05],
                    'subsample':[0.6,0.75],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [64,128,256]
                },
                
                "XGBClassifier":{
                    'learning_rate':[.01,.05],
                    'n_estimators': [64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]
            
            with mlflow.start_run(run_name='first'):
                predict_quality = best_model.predict(X_test)
                
                (f1,recall,precision) = self.eval_metrics(y_test,predict_quality)

                mlflow.log_params(best_params)
                
                mlflow.log_metric("f1_score",f1)
                mlflow.log_metric('recall_score',recall)
                mlflow.log_metric('precision_score',precision)

                mlflow.sklearn.log_model(best_model,'best_model')


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            df_score = accuracy_score(y_test, predicted)
            return df_score



        except Exception as e:
            raise CustomException(e,sys)
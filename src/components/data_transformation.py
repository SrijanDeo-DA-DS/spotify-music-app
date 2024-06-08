from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os,sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from category_encoders import CountEncoder
from src.utils import save_obj

class Feature_Engineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        logging.info("************Feature Engineering Started**************")

    def convert_genre(self,df):
        for i in df:
            if 'pop' in i:
                df[i]= df[i].replace(i,'pop')
            elif 'rock' in i:
                df[i]= df[i].replace(i,'rock')
                
            elif 'hip hop' in i:
                df[i]= df[i].replace(i,'hip hop')
                
            elif 'house' in i:
                df[i]= df[i].replace(i,'house')
                
            elif 'jazz' in i:
                df[i]= df[i].replace(i,'jazz')
                
            elif 'dance' in i:
                df[i]= df[i].replace(i,'dance')
                
            elif 'soul' in i:
                df[i]= df[i].replace(i,'soul')
                
            elif 'contemporary' in i:
                df[i]= df[i].replace(i,'contemporary')
                
            elif 'electronic' in i:
                df[i]= df[i].replace(i,'electronic')
            
    
        #df['Artist Genres'] = df['Artist Genres'].apply(lambda i:convert_genre(i))

    def transform_data(self,df):
        try:

            df.drop(['Track URI','Artist URI(s)','Album URI','Album Artist URI(s)','Album Image URL','Track Preview URL',
                  'ISRC','Added By','Album Genres','Track Number','Track Name','Album Name','Album Artist Name(s)',
                 'Album Release Date','Disc Number','Added At', 'Artist Name(s)'],axis=1,inplace=True)

            #self.top_30_genres(df, 'Artist Genre')

            #df.drop(['Artist Genre'])

            self.convert_genre(df)

            #df['Artist Genres'] = df['Artist Genres'].apply(lambda i:self.convert_genre(i))

            df = df[df['Artist Genres'].isin(df['Artist Genres'].value_counts().nlargest(20).index)]

            logging.info("Dropping & transforming columns from orignal dataset")

            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def fit(self,X,y=None):
        return self
        
    def transform(self, X:pd.DataFrame, y=None):
        try:
            transformed_df = self.transform_data(X)

            return transformed_df
        except Exception as e:
            raise CustomException(e,sys)

@dataclass
class DataTransformationConfig():
    processed_obj_file_path = PREPROCESSING_OBJ_FILE
    transform_train_path = TRANSFORM_TRAIN_FILE_PATH
    transform_test_path = TRANSFORM_TEST_FILE_PATH
    transform_validation_path = TRANSFORM_VALIDATION_FILE_PATH
    feature_engg_obj_path = FEATURE_ENGG_OBJ_FILE_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_columns = ['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Time Signature']
            categorical_columns = ['Artist Genres', 'Label', 'Copyrights']
            one_hot_encoding_columns = ['Explicit']

            # Numerical pipeline --> Use for transforming num fts like yeo-johnson/ missing/outliers etc.
            numerical_pipeline = Pipeline(
                steps = [('impute',SimpleImputer(strategy='mean'))]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps = [('impute', SimpleImputer(strategy='most_frequent')),
                         ('freq_enc', CountEncoder(handle_unknown = -1, min_group_size=50)),
                         ]
            )

            ## One hot encoding
            ohe_pipeline = Pipeline(
                steps = [('one_hot', OneHotEncoder(handle_unknown = 'ignore', sparse_output=False,drop='first').set_output(transform="pandas"))]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns),
                    ('ohe_pipeline', ohe_pipeline, one_hot_encoding_columns)
                ]
            )

            return preprocessor
        
            logging.info("Pipeline steps completed")

        except Exception as e:
            raise CustomException(e,sys)
        
    def get_feature_engineering_obj(self):
        try:
            feature_engineering = Pipeline(
                steps = [('fe', Feature_Engineering())]
            )

            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path, validation_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            validation_df = pd.read_csv(validation_path)

            logging.info("Obtaining FE steps object")

            fe_obj = self.get_feature_engineering_obj()

            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)
            validation_df = fe_obj.transform(validation_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            validation_df.to_csv("validation_data.csv")


            processing_obj = self.get_data_transformation_obj()

            target_columns_name = "Popularity"

            X_train = train_df.drop(columns = target_columns_name,axis=1)
            y_train = train_df[target_columns_name]

            X_test = test_df.drop(columns = target_columns_name,axis=1)
            y_test = test_df[target_columns_name]

            X_validation = validation_df.drop(columns = target_columns_name,axis=1)
            y_validation = validation_df[target_columns_name]

            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)
            X_validation = processing_obj.transform(X_validation)

            train_arr = np.c_[X_train,np.array(y_train)]
            test_arr = np.c_[X_test,np.array(y_test)]
            validation_arr = np.c_[X_validation,np.array(y_validation)]

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)
            df_validation = pd.DataFrame(validation_arr)

            os.makedirs(os.path.join(self.data_transformation_config.transform_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_path, index=False,header=True)

            os.makedirs(os.path.join(self.data_transformation_config.transform_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_path, index=False,header=True)

            os.makedirs(os.path.join(self.data_transformation_config.tran),exist_ok=True)
            df_validation.to_csv(self.data_transformation_config.transform_validation_path, index=False,header=True)

            save_obj(file_path = self.data_transformation_config.processed_obj_file_path,
                     obj = fe_obj)

            save_obj(file_path = self.data_transformation_config.feature_engg_obj_path,
                     obj = fe_obj)
            
            return (train_arr,test_arr,validation_arr,self.data_transformation_config.processed_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)
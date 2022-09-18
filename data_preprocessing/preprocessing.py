from cProfile import label
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

"this class is use for preprocessing"

class Preprocessor:

    """Use for preprocessing the data"""

    def __init__(self,file_object,logger_object):

        self.file_object = file_object

        self.logger_object = logger_object

    def remove_columns(self,data,columns):

        """This function is use to remove unwanted columns from data"""

        self.logger_object.log(self.file_object,'This will remove the unwanted columns!!..')

        self.data = data
        self.columns=columns

        try:

            self.new_data  = self.data.drop(labels = self.columns,axis=1)#this line will remove the unwanted columns
            self.logger_object.log(self.file_object,'Columns Droped successfully!!...')
            return self.new_data

        except Exception as e:

            self.logger_object.log(self.file_object,'There is some error while dropping columns')

            raise Exception()

    def seprate_lable_feature(self,data,label_column_name):

        """This class will seprate data into dependent and independent variable"""

        self.logger_object.log(self.file_object,'We are seprating data into dependent and independent columns')
        self.data = data
        self.label_column_name = label_column_name
        try:
            self.X = self.data.drop(labels = self.label_column_name,axis=1)
            self.Y = self.data[self.label_column_name]
            self.logger_object.log(self.file_object,'We successfully seprate columns into dependent and independent columns!!...')
            return self.X,self.Y

        except Exception as e:

            self.logger_object.log(self.file_object,'There is some error while seprating')
            self.logger_object.log(self.file_object,'The error while seprating is : ' + str(e))

            raise Exception()

    def is_null_present(self,data):

        """This method check null value and if exsist it return True else false"""

        self.logger_object.log(self.file_object,'No we going to check is null present or not')

        self.data = data
        self.null_present = False
        try:
            self.null_counts = self.data.isna().sum()
            for i in self.null_counts:
                if i >= 1:
                    self.null_present = True
                    break
            if(self.null_present):
                dataframe_with_null = pd.DataFrame() # write the logs to see which columns have null values
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv')#storing null count into file

            return self.null_present

        except Exception as e:

            self.logger_object.log(self.file_object,'There is some error while checking null')
            self.logger_object.log(self.file_object,'The error while checking null is '+str(e))
            raise Exception()


    def impute_missing_value(self,data):

        """This will fill null value with any valuable value"""

        self.logger_object.log(self.file_object,'Now we filling nan with any valuable value using KNNImputer')
        self.data = data
        
        try:

            imputer = SimpleImputer(strategy='median',missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(self.new_array,columns=self.data.columns)
            self.logger_object.log(self.file_object,'Nan value handel successfully!!...')
            return self.new_data

        except Exception as e:

            self.logger_object.log(self.file_object,'There is some error while imputing nan value by valuable value!!...')
            self.logger_object.log(self.file_object,'Error while imputing nan is : '+str(e))
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):

        """Now we checking the columns having standard deviation zero we will drop those columns
        because those columns don't have any importance to dependent variable"""

        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')

        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

    def conversion(self, data):

            self.logger_object.log(self.file_object, 'Concerting 1 to 0 and -1 tot 1')
            self.data = data
            try:

                self.data['Output'] = self.data['Output'].map({1:0, -1: 1})

                self.logger_object.log(self.file_object, 'mapping done successfully!!...')

                return self.data

            except Exception as e:

                self.logger_object.log(self.file_object, 'Error while mapping is : ' + str(e))

                raise Exception()
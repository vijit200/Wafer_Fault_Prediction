from sklearn.model_selection import train_test_split
from data_ingestion import data_loder
from data_preprocessing import preprocessing
from data_preprocessing import Clustering
from best_model_finder import tuner
from file_operations import file_methods
from Logging.logger import App_Logger


class trainModel:

    def __init__(self):

        self.logger_writer = App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):

        self.logger_writer.log(self.file_object,'Model Processing Started...')

        #loading data 
        try:
            data_getter = data_loder.Data_Getter(self.file_object,self.logger_writer)

            data = data_getter.get_data()

            """Now doing preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.logger_writer)

            data=preprocessor.remove_columns(data,['Wafer'])#removing columns

            #converting output 1 to 0 and -1 to 1

            data = preprocessor.conversion(data)

            #we seprating dependent and independent variable
            X,Y = preprocessor.seprate_lable_feature(data,label_column_name='Output')

            #check if missing values are present in the dataset

            is_null_present = preprocessor.is_null_present(X)

            #Imputing to missing value

            if(is_null_present):

                X = preprocessor.impute_missing_value(X)

            # Now handling zero standard deviation

            col_to_drop = preprocessor.get_columns_with_zero_std_deviation(X)

            X = preprocessor.remove_columns(X,col_to_drop)


            self.logger_writer.log(self.file_object,'Model Processing Ended Successfully...')


            """Now doing clustering"""


            kmeans = Clustering.KMeansClustering(self.logger_writer,self.file_object)

            number_cluster = kmeans.elbow_plot(X)

            # Divide the data into clusters

            X=kmeans.create_clusters(X,number_cluster)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            list_of_clusters=X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                    cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                    # Prepare the feature and Label columns
                    cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                    cluster_label= cluster_data['Labels']

                    # splitting the data into training and test set for each cluster one by one
                    x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                    model_finder=tuner.model_finder(self.logger_writer,self.file_object) # object initialization

                    #getting the best model for each of the clusters
                    best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                    #saving the best model to the directory.
                    file_op = file_methods.File_Operation(self.file_object,self.logger_writer)
                    save_model=file_op.save_model(best_model,best_model_name+str(i))

                # logging the successful Training
            self.logger_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.logger_writer.log(self.file_object, 'Unsuccessful End of Training  '+str(e) )
            self.file_object.close()
            raise Exception()




        
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:

    """This class will devide my data into n cluster"""

    def __init__(self,logger_object,file_object):
        self.logger_object = logger_object
        self.file_object = file_object

    def elbow_plot(self,data):

        """Here by using kmean and elbow plot we define number of cluster"""

        self.logger_object.log(self.file_object,'Now we are doing clustering using KMeans')

        self.data = data
        wcss = []
        try:
            for i in range(1,11):
                kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
                kmeans.fit(self.data)
                wcss.append(kmeans.inertia_)

            plt.plot(range(1,11),wcss)
            plt.title('Elbow-Plot')
            plt.xlabel('cluster')
            plt.ylabel('wcss')
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')# saving the elbow plot locally
            self.kn = 3 #KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(self.file_object, 'The optimum number of clusters is: '+str(self.kn)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self,data,number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data=data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

            self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') # saving the KMeans model to directory
                                                                                    # passing 'Model' as the functions need three parameters

            self.data['Cluster']=self.y_kmeans  # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score

class model_finder:

    def __init__(self,logger_object,file_object):
        self.logger_object =logger_object
        self.file_object = file_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_random_forest(self,train_x,train_y):

        """ This function use for hyperparameters tuning of random forest"""

        self.logger_object.log(self.file_object,'Now we are doing hyper tuning of random forest')

        try:
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # now using grid search cv 

            self.ram = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)

            self.ram.fit(train_x,train_y)

            self.criterion = self.ram.best_params_['criterion']
            self.n_estimators = self.ram.best_params_['n_estimators']
            self.max_depth = self.ram.best_params_['max_depth']
            self.max_features = self.ram.best_params_['max_features']

            self.clf = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,max_depth=self.max_depth,max_features=self.max_features)

            self.clf.fit(train_x,train_y)

            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.ram.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf

        except Exception as e:

            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_xgboost(self,train_x,train_y):

        """This method use for hyperparameters of xgboost classifier """


        self.logger_object.log(self.file_object,'Now we are doing hyper tuning of Xgboost')

        try:

            self.params = {
                 'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]
            }

            self.grid = GridSearchCV(self.xgb,param_grid=self.params,cv=5,verbose=3)

            self.grid.fit(train_x,train_y)

            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_depth = self.grid.best_params_['max_depth']
            self.learning_rate = self.grid.best_params_['learning_rate']

            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)

            self.xgb.fit(train_x,train_y)


            self.logger_object.log(self.file_object,
                                   'XGBoost best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_XGBoost method of the Model_Finder class')

            return self.xgb

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):

        """ This class help to give best model for prediction """
        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')

        try:
            self.xgboost = self.get_best_params_for_xgboost(train_x,train_y)

            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))

            #comparing the two models
            if(self.random_forest_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()



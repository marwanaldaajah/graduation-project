import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config.database import Database
from config.data_preparer import DataPreparer
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.neighbors import NearestCentroid
# from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from model.model import Model

# # Build machine learning models
# machineLearningModels = [
#     ("Logistic Regression", LogisticRegression(
#         max_iter=1000, solver='liblinear')),
#     ("Decision Tree", DecisionTreeClassifier()),
#     ("Support Vector Machine", SVC()),
#     ("Random Forest", RandomForestClassifier()),
#     ("Ridge Classifier", RidgeClassifier()),
#     ("SGD Classifier", SGDClassifier()),
#     ("Passive Aggressive Classifier", PassiveAggressiveClassifier()),
#     ("Perceptron", Perceptron()),
#     ("Bernoulli Naive Bayes", BernoulliNB()),
#     ("Multinomial Naive Bayes", MultinomialNB()),
#     ("Nearest Centroid", NearestCentroid()),
#     ("Bagging Classifier", BaggingClassifier()),
#     ("Extra Trees Classifier", ExtraTreesClassifier()),
#     ("Gradient Boosting", GradientBoostingClassifier()),
#     ("K-Nearest Neighbors", KNeighborsClassifier()),
#     ("Naive Bayes", GaussianNB()),
#     ("Adaptive Boosting", AdaBoostClassifier()),
#     ("Multi-layer Perceptron", MLPClassifier(max_iter=500)),
#     ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
#     ("Quadratic Discriminant Analysis",
#         QuadraticDiscriminantAnalysis(reg_param=0.1)),
# ]


class MachineLearning:

    def evaluate_models(self):
        db = Database()
        df = db.readbileScv()
        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        # Create data preparer instance
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = preparer.split_data()
        index = 0
        results = []
        # Loop through all machine learning models and compute performance metrics
        for i in range(19):
            try:
                model_path = f"result/{i}.joblib"
                model = joblib.load(model_path)
                model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = self.get_accuracy(y_test, y_pred)
                results.append({
                    'index': index,
                    'name': model,
                    'accuracy': accuracy,
                    'model': model
                })
                index = index + 1
            except Exception as e:
                print(
                    f"Exception occurred while evaluating {index} model. {str(e)}")
                continue
        return results

    # def evaluate_models(self):
    #     db = Database()
    #     df = db.readbileScv()
    #     input_data = db.readInputColumn()
    #     output_data = db.readOutputColumns()
    #     train_size = db.readTrainSize()
    #     test_size = db.readTestSize()
    #     # Create data preparer instance
    #     preparer = DataPreparer(
    #         df, input_data, output_data, train_size, test_size)
    #     # Split data into training and testing sets
    #     X_train, X_test, y_train, y_test = preparer.split_data()
    #     index = 0
    #     results = []
    #     # Loop through all machine learning models and compute performance metrics
    #     for name, model in machineLearningModels:
    #         try:
    #             model = model.fit(X_train, y_train)
    #             y_pred = model.predict(X_test)
    #             accuracy = self.get_accuracy(y_test, y_pred)
    #             results.append({
    #                 'index': index,
    #                 'name': name,
    #                 'accuracy': accuracy,
    #                 'model': model
    #             })
    #             index = index + 1
    #         except Exception as e:
    #             print(
    #                 f"Exception occurred while evaluating {name} model. {str(e)}")
    #             continue
    #     return results

    def evaluate_model(self, model):
        db = Database()
        df = db.readbileScv()
        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        # Create data preparer instance
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = preparer.split_data()
        # Loop through all machine learning models and compute performance metrics

        result = None  # default value
        try:
            y_pred = model.predict(X_test)
            accuracy = self.get_accuracy(y_test, y_pred)
            precision = self.get_precision(y_test, y_pred)
            recall = self.get_recall(y_test, y_pred)
            f1 = self.get_f1(y_test, y_pred)
            result = ({
                'name': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
        except Exception as e:
            print(
                f"Exception occurred while evaluating . {str(e)}")

        return result

    def get_accuracy(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

    def get_precision(self, y_test, y_pred):
        return precision_score(y_test, y_pred, average='macro')

    def get_recall(self, y_test, y_pred):
        return recall_score(y_test, y_pred, average='macro')

    def get_f1(self, y_test, y_pred):
        return f1_score(y_test, y_pred, average='macro')

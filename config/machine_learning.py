import threading
import time
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config.database import Database
from config.data_preparer import DataPreparer
from model.model import Model

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Build machine learning models
machineLearningModells = [
    ("Logistic Regression", LogisticRegression(
        max_iter=1000, solver='liblinear')),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Support Vector Machine", SVC()),
    ("Random Forest", RandomForestClassifier()),
    ("Ridge Classifier", RidgeClassifier()),
    ("SGD Classifier", SGDClassifier()),
    ("Passive Aggressive Classifier", PassiveAggressiveClassifier()),
    ("Perceptron", Perceptron()),
    ("Bernoulli Naive Bayes", BernoulliNB()),
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Nearest Centroid", NearestCentroid()),
    ("Bagging Classifier", BaggingClassifier()),
    ("Extra Trees Classifier", ExtraTreesClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Adaptive Boosting", AdaBoostClassifier()),
    ("Multi-layer Perceptron", MLPClassifier(max_iter=500)),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("Quadratic Discriminant Analysis",
        QuadraticDiscriminantAnalysis(reg_param=0.1)),
]

# Build machine learning models
machineLearningModels = [
    ("Logistic Regression", f"result/{0}.joblib"),
    ("Decision Tree", f"result/{1}.joblib"),
    ("Support Vector Machine", f"result/{2}.joblib"),
    ("Random Forest", f"result/{3}.joblib"),
    ("Ridge Classifier", f"result/{4}.joblib"),
    ("SGD Classifier", f"result/{5}.joblib"),
    ("Passive Aggressive Classifier", f"result/{6}.joblib"),
    ("Perceptron", f"result/{7}.joblib"),
    ("Bernoulli Naive Bayes", f"result/{8}.joblib"),
    ("Multinomial Naive Bayes", f"result/{9}.joblib"),
    ("Nearest Centroid", f"result/{10}.joblib"),
    ("Bagging Classifier", f"result/{11}.joblib"),
    ("Extra Trees Classifier", f"result/{12}.joblib"),
    ("Gradient Boosting", f"result/{13}.joblib"),
    ("K-Nearest Neighbors", f"result/{14}.joblib"),
    ("Naive Bayes", f"result/{15}.joblib"),
    ("Adaptive Boosting", f"result/{16}.joblib"),
    ("Multi-layer Perceptron", f"result/{17}.joblib"),
    ("Linear Discriminant Analysis", f"result/{18}.joblib"),
    ("Quadratic Discriminant Analysis", f"result/{19}.joblib"),
]


class ModelEvaluator(threading.Thread):
    def __init__(self, index, name, model_path, X_train, y_train, X_test, y_test, results):
        super().__init__()
        self.index = index
        self.name = name
        self.model_path = model_path
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = results

    def run(self):
        try:
            model = joblib.load(self.model_path)
            model = model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(
                self.y_test, y_pred, average='macro', zero_division=1)
            recall = recall_score(self.y_test, y_pred,
                                  average='macro', zero_division=1)
            f1 = f1_score(self.y_test, y_pred,
                          average='macro', zero_division=1)
            result_obj = Model(self.index, self.name, accuracy,
                               precision, recall, f1, model, y_pred)
            self.results.append(result_obj)
        except Exception as e:
            print(
                f"Exception occurred while evaluating {self.name} model. {str(e)}")


class MachineLearning:
    def evaluateModels(self):
        start_time = time.time()
        db = Database()
        df = db.readbileScv()
        db.dataDiscribstion()
        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        X_train, X_test, y_train, y_test = preparer.split_data()

        results = []
        threads = []
        for index, (name, model_path) in enumerate(machineLearningModels):
            thread = ModelEvaluator(
                index, name, model_path, X_train, y_train, X_test, y_test, results)
            threads.append(thread)

        for thread in threads:
            thread.start()
            print(
                f"Thread {thread.name} is created and is active: {thread.is_alive()}")
            thread.join()

        end_time = time.time()
        eval_time = end_time - start_time
        print(
            f'measure the time taken to evaluate the models using threading & joblib: {eval_time}')

        return results

    def evaluateModelJoblib(self, model):
        db = Database()
        df = db.readbileScv()
        db.dataDiscribstion()

        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)

        X_train, X_test, y_train, y_test = preparer.split_data()

        result = None
        try:
            y_pred = model.predict(X_test)
            accuracy = self.get_accuracy(y_test, y_pred)
            precision = self.get_precision(y_test, y_pred)
            recall = self.get_recall(y_test, y_pred)
            f1 = self.get_f1(y_test, y_pred)
            result = {
                'name': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        except Exception as e:
            print(f'Exception occurred while evaluating . {str(e)}')
        return result

    def evaluateModelsUsJoblib(self):
        start_time = time.time()
        db = Database()
        df = db.readbileScv()
        db.dataDiscribstion()
        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        X_train, X_test, y_train, y_test = preparer.split_data()
        index = 0
        results = []
        for name, model in machineLearningModels:
            try:
                model = joblib.load(model)
                model = model.fit(X_train, y_train)
                y_pred = self.get_predict(model, X_test)
                accuracy = self.get_accuracy(y_test, y_pred)
                precision = self.get_precision(y_test, y_pred)
                recall = self.get_recall(y_test, y_pred)
                f1 = self.get_f1(y_test, y_pred)
                result_obj = Model(index, name, accuracy, precision, recall, f1, model, y_pred)
                results.append(result_obj)
                index = index + 1
            except Exception as e:
                print(f'Exception occurred while evaluating {name} model. {str(e)}')
                continue
        end_time = time.time()
        eval_time = end_time - start_time
        print(
            f'measure the time taken to evaluate the models us joblib {eval_time}')

        return results

    def evaluateModelsUsFixedLest(self):
        start_time = time.time()
        db = Database()
        df = db.readbileScv()
        db.dataDiscribstion()
        input_data = db.readInputColumn()
        output_data = db.readOutputColumns()
        train_size = db.readTrainSize()
        test_size = db.readTestSize()
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        X_train, X_test, y_train, y_test = preparer.split_data()
        index = 0
        results = []
        for name, model in machineLearningModells:
            try:
                # model_start_time = time.time()
                model = model.fit(X_train, y_train)
                y_pred = self.get_predict(model, X_test)
                accuracy = self.get_accuracy(y_test, y_pred)
                precision = self.get_precision(y_test, y_pred)
                recall = self.get_recall(y_test, y_pred)
                f1 = self.get_f1(y_test, y_pred)
                result_obj = Model(index, name, accuracy,
                                   precision, recall, f1, model, y_pred)
                results.append(result_obj)
                # model_end_time = time.time()
                # model_eval_time = model_end_time - model_start_time
                # print(f'model {name}. time {model_eval_time}')
                index = index + 1
            except Exception as e:
                print(
                    f'Exception occurred while evaluating {name} model. {str(e)}')
                continue
        end_time = time.time()
        eval_time = end_time - start_time
        print(
            f'measure the time taken to evaluate the models use fixed list {eval_time}')

        return results

    def get_fit(self, model, X_train, y_train):
        return model.fit(X_train, y_train)

    def get_predict(self, model, X_test):
        return model.predict(X_test)

    def get_accuracy(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

    def get_precision(self, y_test, y_pred):
        return precision_score(y_test, y_pred, average='macro')

    def get_recall(self, y_test, y_pred):
        return recall_score(y_test, y_pred, average='macro')

    def get_f1(self, y_test, y_pred):
        return f1_score(y_test, y_pred, average='macro')

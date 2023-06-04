import threading
import time
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from config.database import Database
from config.data_preparer import DataPreparer
from model.model import Model

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
        results = []
        index = 0
        for name, model_path in machineLearningModels:
            try:
                result_obj = self.run_madel(
                    index, name, model_path, df, input_data, output_data, train_size, test_size)
                results.append(result_obj)
                index = index + 1
            except Exception as e:
                print(
                    f'Exception occurred while evaluating {name} model. {str(e)}')
                continue
        end_time = time.time()

        print(f'time without threading  = {end_time-start_time}')

        return results

    def run_madel(self, index, name, model_path, df, input_data, output_data, train_size, test_size):
        preparer = DataPreparer(
            df, input_data, output_data, train_size, test_size)
        X_train, X_test, y_train, y_test = preparer.split_data()

        try:
            model = joblib.load(model_path)
            model = model.fit(X_train, y_train)
            y_pred = self.get_predict(model, X_test)
            accuracy = self.get_accuracy(y_test, y_pred)
            precision = self.get_precision(y_test, y_pred)
            recall = self.get_recall(y_test, y_pred)
            f1 = self.get_f1(y_test, y_pred)
            result_obj = Model(index, name, self.format_percentage(accuracy), self.format_percentage(
                precision), self.format_percentage(recall), self.format_percentage(f1), model, y_pred)
            return result_obj
        except Exception as e:
            print(
                f'Exception occurred while evaluating {name} model. {str(e)}')

    def evaluateSpecificMode(self, model):
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
                'accuracy': self.format_percentage(accuracy),
                'precision': self.format_percentage(precision),
                'recall': self.format_percentage(recall),
                'f1': self.format_percentage(f1),
            }
        except Exception as e:
            print(f'Exception occurred while evaluating . {str(e)}')
        return result

    def format_percentage(self, value):
        return '{:.2f}%'.format(value * 100)

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

# import threading
# import time
# import joblib
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from config.database import Database
# from config.data_preparer import DataPreparer
# from model.model import Model

# # Build machine learning models
# machineLearningModels = [
#     ("Logistic Regression", f"result/{0}.joblib"),
#     ("Decision Tree", f"result/{1}.joblib"),
#     ("Support Vector Machine", f"result/{2}.joblib"),
#     ("Random Forest", f"result/{3}.joblib"),
#     ("Ridge Classifier", f"result/{4}.joblib"),
#     ("SGD Classifier", f"result/{5}.joblib"),
#     ("Passive Aggressive Classifier", f"result/{6}.joblib"),
#     ("Perceptron", f"result/{7}.joblib"),
#     ("Bernoulli Naive Bayes", f"result/{8}.joblib"),
#     ("Multinomial Naive Bayes", f"result/{9}.joblib"),
#     ("Nearest Centroid", f"result/{10}.joblib"),
#     ("Bagging Classifier", f"result/{11}.joblib"),
#     ("Extra Trees Classifier", f"result/{12}.joblib"),
#     ("Gradient Boosting", f"result/{13}.joblib"),
#     ("K-Nearest Neighbors", f"result/{14}.joblib"),
#     ("Naive Bayes", f"result/{15}.joblib"),
#     ("Adaptive Boosting", f"result/{16}.joblib"),
#     ("Multi-layer Perceptron", f"result/{17}.joblib"),
#     ("Linear Discriminant Analysis", f"result/{18}.joblib"),
#     ("Quadratic Discriminant Analysis", f"result/{19}.joblib"),
# ]

# lock = threading.Lock()

# class MachineLearning:
#     def evaluateModels(self):
#         start_time = time.time()
#         db = Database()
#         df = db.readbileScv()
#         db.dataDiscribstion()
#         input_data = db.readInputColumn()
#         output_data = db.readOutputColumns()
#         train_size = db.readTrainSize()
#         test_size = db.readTestSize()
#         results = []
#         index = 0
#         threads = []

#         for name, model_path in machineLearningModels:
#             try:
#                 thread = threading.Thread(target=self.run_madel, args=(index, name, model_path, df, input_data, output_data, train_size, test_size, results))
#                 threads.append(thread)
#                 thread.start()
#                 index += 1
#             except Exception as e:
#                 print(f'Exception occurred while evaluating {name} model. {str(e)}')
#                 continue

#         for thread in threads:
#             thread.join()

#         end_time = time.time()
#         print(f'Total time with threading: {end_time - start_time}')

#         return results

#     def run_madel(self, index, name, model_path, df, input_data, output_data, train_size, test_size, results):
#         preparer = DataPreparer(df, input_data, output_data, train_size, test_size)
#         X_train, X_test, y_train, y_test = preparer.split_data()

#         try:
#             model = joblib.load(model_path)
#             model = model.fit(X_train, y_train)
#             y_pred = self.format_percentage(self.get_predict(model, X_test))
#             accuracy = self.format_percentage(self.get_accuracy(y_test, y_pred))
#             precision = self.format_percentage(self.get_precision(y_test, y_pred))
#             recall = self.format_percentage(self.get_recall(y_test, y_pred))
#             f1 = self.format_percentage(self.get_f1(y_test, y_pred))
#             result_obj = Model(index, name, accuracy, precision, recall, f1, model, y_pred)

#             lock.acquire()
#             results.append(result_obj)
#             lock.release()
#         except Exception as e:
#             print(f'Exception occurred while evaluating {name} model. {str(e)}')


#     def evaluateSpecificMode(self, model):
#         db = Database()
#         df = db.readbileScv()
#         db.dataDiscribstion()

#         input_data = db.readInputColumn()
#         output_data = db.readOutputColumns()
#         train_size = db.readTrainSize()
#         test_size = db.readTestSize()
#         preparer = DataPreparer(
#             df, input_data, output_data, train_size, test_size)

#         X_train, X_test, y_train, y_test = preparer.split_data()

#         result = None
#         try:
#             y_pred = model.predict(X_test)
#             accuracy = self.get_accuracy(y_test, y_pred)
#             precision = self.get_precision(y_test, y_pred)
#             recall = self.get_recall(y_test, y_pred)
#             f1 = self.get_f1(y_test, y_pred)
#             result = {
#                 'name': model,
#                 'accuracy': self.format_percentage(accuracy),
#                 'precision': self.format_percentage(precision),
#                 'recall': self.format_percentage(recall),
#                 'f1': self.format_percentage(f1),
#             }
#         except Exception as e:
#             print(f'Exception occurred while evaluating . {str(e)}')
#         return result

#     def format_percentage(self, value):
#         return '{:.2f}%'.format(value * 100)

#     def get_fit(self, model, X_train, y_train):
#         return model.fit(X_train, y_train)

#     def get_predict(self, model, X_test):
#         return model.predict(X_test)

#     def get_accuracy(self, y_test, y_pred):
#         return accuracy_score(y_test, y_pred)

#     def get_precision(self, y_test, y_pred):
#         return precision_score(y_test, y_pred, average='macro')

#     def get_recall(self, y_test, y_pred):
#         return recall_score(y_test, y_pred, average='macro')

#     def get_f1(self, y_test, y_pred):
#         return f1_score(y_test, y_pred, average='macro')

import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

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

app = Flask(__name__)
app.secret_key = 'fefe'

key_file_name = 'file_name'
key_input_columns = 'input_columns'
key_output_columns = 'output_columns'
key_train_size = 'train_size'
key_test_size = 'test_size'
key_model = 'model'
key_scaler = 'key_scaler'
key_accuracy = 'accuracy'


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            print("No file uploaded.")
            flash("No file uploaded.")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            print("No file selected.")
            flash("No file selected.")
            return redirect(request.url)

        file_path = f'uploads/{file.filename}'
        file.save(file_path)

        session[key_file_name] = file_path
        session.pop(key_input_columns, None)
        session.pop(key_output_columns, None)
        session.pop(key_train_size, None)
        session.pop(key_test_size, None)

        return redirect(url_for("select_columns"))

    return render_template("index.html")


@app.route("/select_columns", methods=["GET", "POST"])
def select_columns():
    try:
        df = pd.read_csv(session[key_file_name])
    except KeyError:
        print("Please upload a CSV file first.")
        flash("Please upload a CSV file first.")
        return redirect(url_for("index"))

    columns = df.columns.tolist()

    if request.method == "POST":
        input_columns = request.form.getlist("input_col")
        output_columns = request.form.getlist("output_col")
        if not input_columns or not output_columns:
            print("Please select input and output columns.")
            flash("Please select input and output columns.")
            return redirect(request.url)
        session[key_input_columns] = input_columns
        session[key_output_columns] = output_columns
        return redirect(url_for("split_data"))

    return render_template("select_columns.html", columns=columns)


@app.route("/split_data", methods=["GET", "POST"])
def split_data():
    input_columns = session.get(key_input_columns)
    output_columns = session.get(key_output_columns)

    if not input_columns or not output_columns:
        print("Please select input and output columns first.")
        flash("Please select input and output columns first.")
        return redirect(url_for("select_columns"))

    if request.method == "POST":
        train_size = request.form.get("train_size")
        test_size = request.form.get("test_size")
        if not train_size or not test_size:
            flash("Please enter train and test sizes.")
            return redirect(request.url)
        session[key_train_size] = train_size
        session[key_test_size] = test_size
        return redirect(url_for("train_model"))

    return render_template("split_data.html")


@app.route("/train_model", methods=["GET", "POST"])
def train_model():
    # Show the progress bar
    session['training_in_progress'] = True
    try:
        input_columns = session.get(key_input_columns)
        output_columns = session.get(key_output_columns)
        train_size = float(session.get(key_train_size))
        test_size = float(session.get(key_test_size))
    except KeyError:
        print("Please select train and test sizes first.")
        flash("Please select train and test sizes first.")
        return redirect(url_for("split_data"))

    # Read in the data from the uploaded CSV file
    try:
        df = pd.read_csv(session[key_file_name])
    except KeyError:
        print("Please upload a CSV file first.")
        flash("Please upload a CSV file first.")
        return redirect(url_for("index"))

    # Preprocess the data
    X = df[input_columns]
    y = df[output_columns]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size)
    
    # Flatten y_train and y_test to one-dimensional arrays
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Build machine learning models
    machineLearningModels = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, solver='liblinear')),
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
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ]

    # Train and evaluate models
    models = []
    try:
        for name, model in machineLearningModels:
            print(f"Training model: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # calculate additional metrics as needed
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='micro')
            accuracy = accuracy_score(y_test, y_pred)
            # create a dictionary for this model's results
            model_results = {
                'name': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            models.append(model_results)
    except Exception as e:
        print(f'Error occurred: {e}')
    finally:
        # Hide the progress bar
        session['training_in_progress'] = False

    if request.method == 'POST':
        selectedModel = request.form.get("model")
        if selectedModel is None:
            print("Invalid model selected.")
            flash("Invalid model selected.")
            return redirect(url_for("train_model"))
        session[key_model] = selectedModel
        return redirect(url_for('model_result'))

    return render_template("training.html", models=models)


@app.route("/model_result", methods=["GET", "POST"])
def model_result():
    selectedModel = session.get(key_model)
    print(f'selectedModel: {selectedModel}')
    return render_template("model_results.html", model=selectedModel)


HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)

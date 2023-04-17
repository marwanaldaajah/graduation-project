import json
from database import Database
import os
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
from machine_learning import MachineLearning


app = Flask(__name__, template_folder='templates')
app.secret_key = 'f1F@'


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
    return render_template("index.html")


@app.route('/select_columns', methods=["GET", "POST"])
def select_columns():
    db = Database()
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            print('Please select file.')
            return redirect(url_for('index'))
        else:
            file_path = db.saveFile(file)
            file.save(file_path)

    columns = db.readbileColumns()
    if columns is False:
        return redirect(request.url)
    else:
        return render_template('select_columns.html', columns=columns)


@app.route("/split_data", methods=["GET", "POST"])
def split_data():
    db = Database()
    if request.method == "POST":
        input_columns = request.form.getlist("input_col")
        output_columns = request.form.getlist("output_col")
        if not input_columns or not output_columns:
            print("Please select input and output columns first.")
            return redirect(url_for("select_columns"))
        else:
            db.saveInput(input_columns)
            db.saveOutput(output_columns)

    return render_template("split_data.html")


@app.route("/train_model", methods=["GET", "POST"])
def train_model():
    db = Database()
    if request.method == "POST":
        train_size = request.form.get("train_size")
        test_size = request.form.get("test_size")
        if not train_size or not test_size:
            print('select train and testing size')
            return redirect(url_for("split_data"))
    db.saveTrainSize(train_size)
    db.saveTestSize(test_size)

    machineLearning = MachineLearning()
    models = machineLearning.evaluate_models()
    return render_template("training.html", models=models)


@app.route("/model_result", methods=["GET", "POST"])
def model_result():
    if request.method == 'POST':
        model = request.form.get("model")
    return render_template("model_results.html", model=model)


@app.route('/save_model', methods=["GET", "POST"])
def save_model_route():
    if request.method == 'POST':
        model = request.form.get("model")
        index=model[6]
        print(f'index: {index}')
        # joblib.dump(model, 'model-result.joblib')
        # predictions = model.predict('')
    return render_template("model_results.html", model=model)


HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)


# @app.route("/train_model", methods=["GET", "POST"])
# def train_model():
#     db = Database()
#     if request.method == "POST":
#         train_size = request.form.get("train_size")
#         test_size = request.form.get("test_size")
#         if not train_size or not test_size:
#             print('select train and testing size')
#             return redirect(url_for("split_data"))
#         else:
#             db.saveTestSize(train_size)
#             db.saveTestSize(test_size)

#     df = db.readFileScv()
#     input_columns = db.readInputColumn()
#     output_columns = db.readOutputColumns()
#     train_size = db.readTrainSize()
#     test_size = db.readTestSize()

#     # Handle missing values
#     df.dropna(inplace=True)
#     # print(f'df: {df}')
#     # Convert categorical data to numerical data
#     cat_columns = df.select_dtypes(include=['object']).columns
#     df[cat_columns] = df[cat_columns].apply(
#         lambda x: x.astype('category').cat.codes)
#     # print(f'df: {df}')

#     # Preprocess the data
#     X = df[input_columns]
#     y = df[output_columns]

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, train_size=train_size, test_size=test_size, random_state=42)

#     # Normalize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Check if there are negative values in X_train and handle them
#     if np.any(X_train < 0).any():
#         # Add the minimum value of X_train to make all values non-negative
#         X_train += abs(np.min(X_train))

#     # Check if there are negative values in X_test and handle them
#     if np.any(X_test < 0).any():
#         # Add the minimum value of X_test to make all values non-negative
#         X_test += abs(np.min(X_test))

#     # Flatten y_train and y_test to one-dimensional arrays
#     y_train = np.ravel(y_train)
#     y_test = np.ravel(y_test)

#     # Train and evaluate models
#     models = []
#     i = 0
#     try:
#         for name, model in machineLearningModels:
#             print(f'i: {i} , test model name: {name}')
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             # calculate additional metrics as needed
#             precision = precision_score(y_test, y_pred, average='micro')
#             recall = recall_score(y_test, y_pred, average='micro')
#             f1 = f1_score(y_test, y_pred, average='micro')
#             accuracy = accuracy_score(y_test, y_pred)
#             # create a dictionary for this model's results
#             model_results = {
#                 'index': i,
#                 'name': name,
#                 'accuracy': accuracy,
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1
#             }
#             models.append(model_results)
#             i = i + 1
#     except Exception as e:
#         print(f'Error occurred: {e} model: {name}')

#     return render_template("training.html", models=models)

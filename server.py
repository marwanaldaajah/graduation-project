import pickle
from flask import Flask, render_template, request, redirect, session
import os
from database import Database
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = 'my_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(f'uploads/{filename}')
    return redirect(f'/detect?filename={filename}')


@app.route('/detect', methods=['GET', 'POST'])
def detect():

    filename = request.args.get('filename')
    file_path = f'uploads/{filename}'

    database = Database(file_path)
    df = database.readFileScv()
    columns = df.columns.tolist()

    if request.method == 'POST':
        input_columns = request.form.getlist('input_columns')
        output_columns = request.form.getlist('output_columns')
        if not input_columns or not output_columns:
            return render_template('error.html', message="Please select at least one input and one output column")

    return render_template('detect.html',columns=columns)


@app.route('/result', methods=['POST'])
def result():
    filename = request.args.get('filename')
    file_path = f'uploads/{filename}'

    # Read CSV file
    database = Database(file_path)
    df = database.readFileScv()

    input_columns = request.form.getlist('input_columns')
    output_columns = request.form.getlist('output_columns')
    if not input_columns or not output_columns:
        return render_template('error.html', message="Please select at least one input and one output column")

    # Split data into train and test sets
    X = df[input_columns]
    y = df[output_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return render_template('result.html', input_columns=input_columns, output_columns=output_columns)


HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)

# @app.route('/run-algorithm', methods=['POST'])
# def run_algorithm():
#     splitting_value = request.form['splitting_value']
#     X_train = session.get('X_train')
#     y_train = session.get('y_train')
#     # apply the machine learning algorithm
#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)
#     # save the resulting model for later use
#     model_filename = f'model_{session.sid}.pkl'
#     with open(model_filename, 'wb') as file:
#         pickle.dump(model, file)
#     session['model_filename'] = model_filename
#     return render_template('step4.html', model=model)
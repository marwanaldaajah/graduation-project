from flask import Flask, redirect, render_template, request
import os
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from database import Database

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('step1.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(f'uploads/{filename}')
    return redirect(f'/step2?filename={filename}')


@app.route('/step2', methods=['GET', 'POST'])
def display_table():
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
        return redirect('/step3', file_path=file_path, input_columns=input_columns, output_columns=output_columns)
    

    return render_template('step2.html', columns=columns)


@app.route('/step3', methods=['GET', 'POST'])
def splitting():
    if request.method == 'POST':
        file_path = request.args.get('file_path')
        input_columns = request.form.getlist('input_columns')
        output_columns = request.form.getlist('output_columns')
        
        input1 = request.form.get('input1')
        input2 = request.form.get('input2')

    file = 'uploads/organization.csv'
    database = Database(file)
    df = database.readFileScv()

    X=df[input_columns].values
    y=df[output_columns].values

    print(f'x={X}')
    print(f'y={y}')


    return render_template('step3.html')




HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)


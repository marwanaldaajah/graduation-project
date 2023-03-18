from flask import Flask, render_template, request, redirect
import pandas as pd
import os

app = Flask(__name__)


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

    # Read the CSV data into a Pandas DataFrame
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
    except FileNotFoundError:
        return render_template('error.html', message="File not found")
    except pd.errors.ParserError:
        return render_template('error.html', message="Error parsing the CSV file")

    if request.method == 'POST':
        input_columns = request.form.getlist('input_columns')
        output_columns = request.form.getlist('output_columns')
        if not input_columns or not output_columns:
            return render_template('error.html', message="Please select at least one input and one output column")
        orgs = []
        for index, row in df.iterrows():
            inputs = {}
            outputs = {}
            for col in input_columns:
                inputs[col] = row.get(col)
            for col in output_columns:
                outputs[col] = row.get(col)
            orgs.append({
                'Inputs': inputs,
                'Outputs': outputs
            })
        return render_template('result.html', orgs=orgs)

    return render_template('detect.html', columns=columns)


@app.route('/result', methods=['POST'])
def result():
    input_columns = request.form.getlist('input_columns')
    output_columns = request.form.getlist('output_columns')
    return render_template('result.html', input_columns=input_columns, output_columns=output_columns)


HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)

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


@app.route('/detect')
def detect():
    filename = request.args.get('filename')
    file_path = f'uploads/{filename}'

    # Read the CSV data into a Pandas DataFrame
    try:
        df = pd.read_csv(file_path)
        print(df)
    except FileNotFoundError:
        return render_template('error.html', message="File not found")
    except pd.errors.ParserError:
        return render_template('error.html', message="Error parsing the CSV file")

    orgs = []
    for index, row in df.iterrows():
        name = row.get('Name')
        website = row.get('Website')
        country = row.get('Country')
        if name and website and country:
            orgs.append({
                'Name': name,
                'Website': website,
                'Country': country
            })

    return render_template('detect.html', orgs=orgs)


HOST = 'localhost'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)

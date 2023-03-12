from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    inputData = pd.read_csv(file)
    return render_template('detect.html', inputData=inputData)


@app.route('/result', methods=['POST'])
def result():
    result = request.form['result']
    return render_template('result.html', result=result)


HOST='localhost'
PORT=8000
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(host=HOST,port=PORT)

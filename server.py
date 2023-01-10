from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect',methods=['POST'])
def detect():
    # Add input validation
    inputData = request.form['inputData']
    # Pass result to template
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

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
insertList=[]
inputList=[]
outputList=[]

## index page

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/addToList',methods=['POST'])
def addToLest():
    inputData = request.form['inputData']
    insertList.append(inputData)
    return render_template('index.html')

## detect page

@app.route('/detect',methods=['POST'])
def detect():
    return render_template('detect.html', insertList=insertList)

@app.route('/input', methods=['POST'])
def add_to_input_list():
    input_data = request.get_json()
    inputList.append(input_data['item'])
    print(f'Add Input List: {inputList}')
    return jsonify(inputList)

@app.route('/output', methods=['POST'])
def add_to_output_list():
    output_data = request.get_json()
    outputList.append(output_data['item'])
    print(f'Add Output List: {outputList}')
    return jsonify(outputList)


## result page

@app.route('/result', methods=['POST'])
def result():
    result = request.form['result']
    return render_template('result.html', inputList=inputList,outputList=outputList)


HOST='localhost'
PORT=8000
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(host=HOST,port=PORT)

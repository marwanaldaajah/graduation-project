
from matplotlib import pyplot as plt
from config.database import Database
import os
from flask import Flask, render_template, request, redirect, url_for
import joblib
from config.machine_learning import MachineLearning


app = Flask(__name__, template_folder='templates')
app.secret_key = 'f1F@'


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
    if request.method == "POST":
        train_size = request.form.get("train_size")
        test_size = request.form.get("test_size")
        if not train_size or not test_size:
            print('select train and testing size')
            return redirect(url_for("split_data"))
        db = Database()
        db.saveTrainSize(train_size)
        db.saveTestSize(test_size)

        print('start')
        machineLearning = MachineLearning()
        models = machineLearning.evaluateModels()
        m=machineLearning.evaluateModelsWithThreading()
        print('end')
        if not os.path.exists("result"):
            os.makedirs("result")
        for model in models:
            if model != None:
                joblib.dump(model.model, f"result/{model.index}.joblib")
        return render_template("training.html", models=models)
    return redirect(url_for("split_data"))


@app.route("/model_result", methods=["GET", "POST"])
def model_result():

        index = request.form.get("model")
        model_path = f"result/{index}.joblib"
        load_model = joblib.load(model_path)

        machineLearning = MachineLearning()

        model = machineLearning.evaluateSpecificMode(load_model)
        plot_data = plot_results(model)

        return render_template("model_results.html", model=model, plot_data=plot_data)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def plot_results(model):
    # Extract model name, metrics, and scores from the model dictionary
    model_name = str(model['name'])
    accuracy = model['accuracy']
    precision = model['precision']
    recall = model['recall']
    f1_score = model['f1']

    # Create bar plot
    fig, ax = plt.subplots()
    ax.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1_score])
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} Performance')

    # Save plot to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Encode the buffer as a base64 string and embed it in the HTML page
    plot_data = base64.b64encode(buf.getvalue()).decode('ascii')
    return plot_data



HOST = '0.0.0.0'
PORT = 8000
if not os.path.exists('uploads'):
    os.makedirs
if __name__ == '__main__':
    print(f"[STARTING] server is starting on PORT: {PORT}")
    app.run(debug=True, host=HOST, port=PORT)
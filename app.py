# app.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/search')
def search():
  return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
  query = request.form['query']
  results = do_search(query)
  return render_template('results.html', query=query, results=results)

def do_search(query):
  # Perform the search and return the results
  return ['Result 1', 'Result 2', 'Result 3']

if __name__ == '__main__':
  app.run()
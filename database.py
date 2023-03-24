from flask import render_template
import pandas as pd


class Database:
    def __init__(self, file):
        self.file = file

    def readFileScv(self):
        try:
            return pd.read_csv(self.file)
        except FileNotFoundError:
            return render_template('error.html', message="File not found")
        except pd.errors.ParserError:
            return render_template('error.html', message="Error parsing the CSV file")

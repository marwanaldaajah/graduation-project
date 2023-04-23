from flask import session
import pandas as pd


class Database:
    key_file_name = 'file_name'
    key_input_columns = 'input_columns'
    key_output_columns = 'output_columns'
    key_train_size = 'train_size'
    key_test_size = 'test_size'

    def saveInput(self, input_columns):
        try:
            if self.key_input_columns in session:
                session.pop(self.key_input_columns)
            session[self.key_input_columns] = input_columns
            return True
        except KeyError:
            print("Error occurred when saving input.")
            return False

    def saveOutput(self, output_columns):
        try:
            if self.key_output_columns in session:
                session.pop(self.key_output_columns)
            session[self.key_output_columns] = output_columns
            return True
        except KeyError:
            print("Error occurred when saving output.")
            return False

    def readInputColumn(self):
        try:
            input_columns = session.get(self.key_input_columns)
            return input_columns
        except KeyError:
            print("Error occurred when reading input.")
            return False

    def readOutputColumns(self):
        try:
            output_columns = session.get(self.key_output_columns)
            return output_columns
        except KeyError:
            print("Error occurred when saving output.")
            return False

    def saveFile(self, file):
        try:
            if self.key_file_name in session:
                session.pop(self.key_file_name)
            file_path = f'uploads/{file.filename}'
            session[self.key_file_name] = file_path
            return file_path
        except KeyError:
            print("Error occurred when saving file.")
            return False

    def readInput(self):
        try:
            input_columns = session.get(self.key_input_columns)
            return input_columns
        except KeyError:
            print("Error occurred when reading input.")
            return False

    def readOutput(self):
        try:
            output_columns = session.get(self.key_output_columns)
            return output_columns
        except KeyError:
            print("Error occurred when reading output.")
            return False

    def readbileScv(self):
        try:
            file_path=session.get(self.key_file_name)
            db = pd.read_csv(file_path)
            return db
        except FileNotFoundError as e:
            print(f"Error occurred when reading file. error: {e}")
            return False
        except pd.errors.ParserError as e:
            print(f"Error occurred when reading file. error: {e}")
            return False

    def dataDiscribstion(self):
        try:
            file_path=session.get(self.key_file_name)
            db = pd.read_csv(file_path)
            num_rows, num_cols = db.shape
            print(f'num_rows: {num_rows}')
            print(f'num_cols: {num_cols}')
            return num_rows, num_cols
        except FileNotFoundError as e:
            print(f"Error occurred when reading file. error: {e}")
            return False, False
        except pd.errors.ParserError as e:
            print(f"Error occurred when reading file. error: {e}")
        return False, False


    def readbileColumns(self):
        try:
            db = self.readbileScv()
            columns = db.columns.tolist()
            return columns
        except KeyError as e:
            print(f"Error occurred while getting file columns. error: {e}")
            return False

    def saveTrainSize(self, train_size):
        try:
            if self.key_train_size in session:
                session.pop(self.key_train_size)
            session[self.key_train_size] = train_size
            return True
        except KeyError as e:
            print(f"Error occurred when saving train_size. error: {e}")
            return False

    def saveTestSize(self, test_size):
        try:
            if self.key_test_size in session:
                session.pop(self.key_test_size)
            session[self.key_test_size] = test_size
            return True
        except KeyError as e:
            print(f"Error occurred when saving test_size. error: {e}")
            return False

    def readTrainSize(self):
        try:
            size = float(session.get(self.key_train_size))
            train_size = size
            return train_size
        except KeyError as e:
            print(f"Error occurred when reading train_size. error: {e}")
            return False

    def readTestSize(self):
        try:
            size = float(session.get(self.key_test_size))
            train_size = size
            return train_size
        except KeyError as e:
            print(f"Error occurred when reading test_size. error: {e}")
            return False

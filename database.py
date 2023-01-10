import pandas as pd

class Database:
  def __init__(self, file):
    self.file = file

  def insert_row(self, row):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(self.file)

    # Insert the new row into the DataFrame
    df = df.append(row, ignore_index=True)

    # Write the updated DataFrame to the CSV file
    df.to_csv(self.file, index=False)

  def delete_row(self, index):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(self.file)

    # Delete the specified row from the DataFrame
    df = df.drop(df.index[index])

    # Write the updated DataFrame to the CSV file
    df.to_csv(self.file, index=False)

  def query(self, column, value):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(self.file)

    # Query the DataFrame and return the results
    return df.query(f'{column} == {value}')

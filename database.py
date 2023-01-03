import pandas as pd

def insert_row(file, row):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(file)

  # Insert the new row into the DataFrame
  df = df.append(row, ignore_index=True)

  # Write the updated DataFrame to the CSV file
  df.to_csv(file, index=False)

def delete_row(file, index):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(file)

  # Delete the specified row from the DataFrame
  df = df.drop(df.index[index])

  # Write the updated DataFrame to the CSV file
  df.to_csv(file, index=False)

def query(file, column, value):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(file)

  # Query the DataFrame and return the results
  return df.query(f'{column} == {value}')

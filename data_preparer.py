import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreparer:

    def __init__(self, df, input_data, output_data, train_size, test_size):
        self.df = df
        self.input_data = input_data
        self.output_data = output_data
        self.train_size = train_size
        self.test_size = test_size

    def preprocess_input(self):
        # Preprocess text input data

        # Handle missing values
        self.df.dropna(inplace=True)
        # print(f'db: {db}')

        # Convert categorical data to numerical data
        cat_columns = self.df.select_dtypes(include=['object']).columns
        self.df[cat_columns] = self.df[cat_columns].apply(
            lambda x: x.astype('category').cat.codes)
        # Preprocess the data
        X = self.df[self.input_data]

        return X

    def preprocess_output(self):
        # Preprocess output data
        # Handle missing values
        self.df.dropna(inplace=True)
        # print(f'db: {db}')

        # Convert categorical data to numerical data
        cat_columns = self.df.select_dtypes(include=['object']).columns
        self.df[cat_columns] = self.df[cat_columns].apply(
            lambda x: x.astype('category').cat.codes)
        # Preprocess the data

        Y = self.df[self.output_data]

        return Y

    def split_data(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.preprocess_input(),
                                                            self.preprocess_output(),
                                                            test_size=self.test_size,
                                                            train_size=self.train_size)
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Check if there are negative values in X_train and handle them
        if np.any(X_train < 0).any():
            # Add the minimum value of X_train to make all values non-negative
            X_train += abs(np.min(X_train))

        # Check if there are negative values in X_test and handle them
        if np.any(X_test < 0).any():
            # Add the minimum value of X_test to make all values non-negative
            X_test += abs(np.min(X_test))
        # Flatten y_train and y_test to one-dimensional arrays
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        return X_train, X_test, y_train, y_test

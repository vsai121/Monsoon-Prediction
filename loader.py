import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import csv

INPUT_SIZE = 1
NUM_STEPS = 40 #  DAYS USED TO MAKE PREDICTION
LEAD_TIME = 15  # PREDICITNG LEAD_TIME DAYS AHEAD
TRAIN_TEST_RATIO = 0.09
TRAIN_VALIDATION_RATIO = 0.12

def read_csv_file(filename):
    name = filename

    """initializing the rows list"""
    rows = []

    """reading csv file"""
    with open(name, 'r') as csvfile:
        """creating a csv reader object"""
        csvreader = csv.reader(csvfile)


        """extracting each data row one by one"""
        for row in csvreader:
            rows.append(row)

        """get total number of rows"""
        print("Total no. of rows: %d"%(csvreader.line_num))

    """printing first 5 rows"""
    print('\nFirst 5 rows are:\n')
    for row in rows[:5]:

        for col in row:
            print("%10s"%col),
        print('\n')


    return rows


def read_rainfall():
    data = read_csv_file('daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:
            rainfall.append(float(col))

    return rainfall

def normalize_seq(seq):

    #print(seq)
    seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i-1][-1] - 1.0 for i, curr in enumerate(seq[1:])]
    return seq

def split_data(input):

    """
    Splits a sequence into windows
    """

    seq = [np.array(input[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input) // INPUT_SIZE)]

    #Normalizing seq
    seq = normalize_seq(seq)

    """Split into groups of num_steps"""
    X = np.array([seq[i: i + NUM_STEPS] for i in range(len(seq) - NUM_STEPS - LEAD_TIME)])
    y = np.array([seq[i + NUM_STEPS + LEAD_TIME] for i in range(len(seq) - NUM_STEPS - LEAD_TIME)])

    return X , y


def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    train_size = int(len(X) * (1.0 - TRAIN_TEST_RATIO))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_size = int(len(X_train) * (1- TRAIN_VALIDATION_RATIO))
    X_train , X_validation = X_train[:train_size] , X_train[train_size:]
    y_train, y_validation = y_train[:train_size], y_train[train_size:]

    return X_train , y_train , X_validation , y_validation , X_test , y_test


def process():
    rainfall = read_rainfall()
    #print("Rainfall" , rainfall[0:124])
    plt.plot(rainfall)
    plt.show()
    X,y = split_data(rainfall)
    """
    print(X[0])
    print(y[0])
    print("\n\n\n")

    print(X[1])
    print(y[1])
    print("\n\n\n")

    print(X[2])
    print(y[2])
    print("\n\n\n")

    print(X[-2])
    print(y[-2])
    print("\n\n\n")

    print(X[-1])
    print(y[-1])
    print("\n\n\n")

    """

    X_train , y_train , X_validation , y_validation , X_test , y_test = train_test_split(X,y)

    """
    print(X_train.shape)
    print(y_train.shape)

    print(X_validation.shape)
    print(y_validation.shape)

    print(X_test.shape)
    print(y_test.shape)

    """

    return X_train , y_train , X_validation , y_validation , X_test , y_test

if __name__ == '__main__':
    process()

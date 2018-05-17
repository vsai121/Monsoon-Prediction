import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import csv

import math

INPUT_SIZE = 1
NUM_STEPS =  215#DAYS USED TO MAKE PREDICTION
LEAD_TIME = 29# PREDICITNG LEAD_TIME DAYS AHEAD
TRAIN_TEST_RATIO = 0.08
TRAIN_VALIDATION_RATIO = 0.04



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

    """printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows[:5]:

        for col in row:
            print("%10s"%col),
        print('\n')
    """

    return rows


def read_rainfall():
    data = read_csv_file('Data/daily_rainfall_central_India_1901_2014.csv')
    rainfall = []

    prev_col=0
    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            if float(col)>=5000:
                rainfall.append(float(prev_col)+1)
                prev_col = float(prev_col)

            else:
                rainfall.append(float(col)+1)
                prev_col = float(col)

    return rainfall

def normalize_seq(seq):

    #print(seq)

    seq = [math.log((curr / seq[0]) , 2)/11 for curr in seq]
    return seq

def split_data(input):

    """
    Splits a sequence into windows
    """

    seq = [np.array(input[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input) // INPUT_SIZE)]


    #Normalizing seq
    #seq = normalize_seq(seq)

    X=[]
    y=[]
    y_org=[]

    for i in range(len(seq) - NUM_STEPS - LEAD_TIME):

        temp = np.array(seq[i: i + NUM_STEPS+LEAD_TIME])
        temp1 = normalize_seq(temp)
        X.append(temp1[0:NUM_STEPS])
        y.append(temp1[NUM_STEPS:NUM_STEPS+LEAD_TIME])
        y_org.append(temp[0])
    """
    print(X[0])
    print(y[0])


    print(X[1])
    print(y[1])


    print(X[2])
    print(y[2])



    print(np.min(X))
    print(np.max(X))

    print(np.min(y))
    print(np.max(y))
    """
    
    X = np.asarray(X , dtype=np.float32)
    y = np.asarray(y , dtype=np.float32)
    return X , y , y_org


def train_test_split(X , y , y_org):

    """
    Splitting data into training and test data"
    """

    train_size = int(len(X) * (1.0 - TRAIN_TEST_RATIO))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test , y_org_test = y[:train_size], y[train_size:] , y_org[train_size:]

    train_size = int(len(X_train) * (1- TRAIN_VALIDATION_RATIO))
    X_train , X_validation = X_train[:train_size] , X_train[train_size:]
    y_train, y_validation , y_org_validation = y_train[:train_size], y_train[train_size:] , y_org[train_size:]

    return X_train , y_train , X_validation , y_validation , X_test , y_test , y_org_validation , y_org_test


def process():
    rainfall = read_rainfall()
    #print("Rainfall" , rainfall[0:124])
    plt.plot(rainfall)
    plt.show()
    X,y , y_org = split_data(rainfall)




    y = np.reshape(y , [y.shape[0] , LEAD_TIME])
    X = np.reshape(X , [X.shape[0] , X.shape[1] , 1])

    """
    print(X.shape)
    print(y.shape)

    print(np.max(X))
    print(np.min(X))

    print(np.max(y))
    print(np.min(y))
    """

    """
    print(X[0])
    print(y[0])

    print(X[1])
    print(y[1])
    """
    X_train , y_train , X_validation , y_validation , X_test , y_test , y_org_validation , y_org_test = train_test_split(X,y , y_org)

    """
    print(X_train.shape)
    print(y_train.shape)

    print(X_validation.shape)
    print(y_validation.shape)

    print(X_test.shape)
    print(y_test.shape)

    """
    #print(y_org_validation[0:5])

    return X_train , y_train , X_validation , y_validation , X_test , y_test , y_org_validation , y_org_test

if __name__ == '__main__':
    process()

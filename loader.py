import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
import csv

import math

INPUT_SIZE = 1
NUM_STEPS = 40#DAYS USED TO MAKE PREDICTION
LEAD_TIME = 5# PREDICITNG LEAD_TIME DAYS AHEAD
TRAIN_TEST_RATIO = 0.1
TRAIN_VALIDATION_RATIO = 0.07



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


def normalise_list(raw):
    norm = [float(i)/sum(raw) for i in raw]
    norm = [float(i)/max(raw) for i in raw]

    return norm

def normalize_seq(seq):

    normalised_seq=[]
    for inp in seq:
        temp=[]
        for j in range(inp.shape[0]):

            temp.append(inp[j]/seq[0][j])

        normalised_seq.append(temp)
    return normalised_seq

def read_rainfall():
    data = read_csv_file('Data/Rainfall/daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append(float(col)/35 + 1)

    #rainfall = normalise_list(rainfall)
    return rainfall


def read_slp():
    data = read_csv_file('Data/SLP/daily_slp_central_India_1948_2014.csv')
    slp = []

    prev_col=0
    """Creating list of SLP data"""
    for row in (data):
        for col in row:

            slp.append(float(col)/10000)
    #slp = normalise_list(slp)
    return slp

def read_uwind():
    data = read_csv_file('Data/Uwind/daily_uwnd_central_India_1948_2014.csv')
    uwnd = []

    """Creating list of Uwind data"""
    for row in (data):
        for col in row:

            uwnd.append(float(col)+6)

    #uwnd = normalise_list(uwnd)
    return uwnd

def read_vwind():
    data = read_csv_file('Data/Vwind/daily_vwnd_central_India_1948_2014.csv')
    vwnd = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            vwnd.append(float(col)+8)

    #vwnd = normalise_list(vwnd)
    return vwnd

def read_at():
    data = read_csv_file('Data/AT/daily_at_central_India_1948_2014.csv')
    at = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            at.append(float(col) - 273.15)

    #vwnd = normalise_list(vwnd)
    return at




def split_data(input1 , input2 , input3 , input4 , input5, output):

    """
    Splits a sequence into windows
    """

    seq1 = [np.array(input1[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input1) // INPUT_SIZE)]

    seq2 = [np.array(input2[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input2) // INPUT_SIZE)]

    seq3 = [np.array(input3[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input3) // INPUT_SIZE)]

    seq4 = [np.array(input4[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input4) // INPUT_SIZE)]

    seq5 = [np.array(input5[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input5) // INPUT_SIZE)]

    """
    print(seq1[0:10])
    print(seq2[0:10])
    print(seq3[0:10])
    print(seq4[0:10])

    """

    seq = []

    for i in range(len(seq1)):
        temp=[]
        temp.append(seq1[i])
        temp.append(seq2[i])
        temp.append(seq3[i])
        temp.append(seq4[i])
        temp.append(seq5[i])
        seq.append(temp)

    #print(seq[0:10])

    X=[]
    y=[]
    y_org=[]


    for i in range(len(seq) - NUM_STEPS - LEAD_TIME):

        z = []
        temp = np.array(seq[i: i + NUM_STEPS+LEAD_TIME])
        temp1 = normalize_seq(temp)
        #print(temp1)
        X.append(temp1[0:NUM_STEPS])

        for j in range(LEAD_TIME):
            z.append(temp1[NUM_STEPS+j][0])
        y.append(z)
        #print("Y" , y)
    X = np.asarray(X , dtype=np.float32)
    y = np.asarray(y , dtype=np.float32)

    return X , y



def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    train_size = int(len(X) * (1.0 - TRAIN_TEST_RATIO))

    X_train, X_test = X[:train_size], X[train_size:]

    y_train, y_test  = y[:train_size], y[train_size:]

    train_size = int(len(X_train) * (1- TRAIN_VALIDATION_RATIO))
    X_train , X_validation = X_train[:train_size] , X_train[train_size:]
    y_train, y_validation  = y_train[:train_size], y_train[train_size:]

    return X_train , y_train , X_validation , y_validation , X_test , y_test


def process():
    rainfall = read_rainfall()
    slp = read_slp()
    uwind = read_uwind()
    vwind = read_vwind()
    at = read_at()

    """
    print(len(slp))
    print(len(uwind))
    print(len(vwind))
    """

    X,y  = split_data(rainfall,slp,uwind ,vwind,at,rainfall)


    y = np.reshape(y , [y.shape[0] , LEAD_TIME])
    X = np.reshape(X , [X.shape[0] , X.shape[1] , 5])

    print(X.shape)
    print(y.shape)


    #print(X[0])
    #print(X[1])
    #print(y[0])
    #print(y[1])
    X_train , y_train , X_validation , y_validation , X_test , y_test  = train_test_split(X,y)

    print(X_train.shape)
    print(y_train.shape)
    return X_train , y_train , X_validation , y_validation , X_test , y_test




if __name__ == '__main__':
    process()

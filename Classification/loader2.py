import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn import preprocessing
import csv

import math

INPUT_SIZE = 1
NUM_STEPS = 5#DAYS USED TO MAKE PREDICTION
LEAD_TIME = 1# PREDICITNG LEAD_TIME DAYS AHEAD
TRAIN_TEST_RATIO = 0.1
TRAIN_VALIDATION_RATIO = 0.07
INPUTS = 13


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
    norm = preprocessing.scale(raw)

    return norm

def normalize_seq(seq):


    normalised_seq=[]
    temp=[[] for i in range(len(seq[0]))]
    for j in range(len(seq[0])):
        for inp in seq:
            temp[j].append(inp[j])

        temp[j] = normalise_list(temp[j])
        #print(temp[j])

    for j in range(len(temp[0])):
        norm_temp=[]
        for k in range(len(temp)):
            norm_temp.append(temp[k][j])

        normalised_seq.append(norm_temp)
    return normalised_seq

def one_hot_encode(x):

    if(x==0):
        return [1 , 0 , 0]

    if(x==1):
        return [0 , 1 , 0]

    if(x==2):
        return [0 , 0 , 1]

def read_rainfall():
    data = read_csv_file('..Data/Rainfall/daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append(float(col))

    rainfall = normalise_list(rainfall)
    return rainfall


def read_rainfall_class():
    data = read_csv_file('..Data/Rainfall/class_daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append(one_hot_encode(int(col)))

    #rainfall = normalise_list(rainfall)
    return rainfall



def filename(f):
    if(f==1):
        return '_central_India_'

    if (f==2):
        return '_south_India_'

    if(f==3):
        return '_BOB_'

    if(f==4):
        return '_AS_'


def read_uwind(fileNum):

    base = '..Data/Uwind/daily_uwnd'
    middle = filename(fileNum)
    end = '1948_2014.csv'

    fileName = base + middle + end

    data = read_csv_file(fileName)
    uwnd = []

    """Creating list of Uwind data"""
    for row in (data):
        for col in row:

            uwnd.append(float(col))

    uwnd = normalise_list(uwnd)
    return uwnd


def read_vwind(fileNum):

    base = '..Data/Vwind/daily_vwnd'
    middle = filename(fileNum)
    end = '1948_2014.csv'

    fileName = base + middle + end
    data = read_csv_file(fileName)
    vwnd = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            vwnd.append(float(col))

    vwnd = normalise_list(vwnd)
    return vwnd


def read_at(fileNum):

    base = "..Data/AT/daily_at"
    middle = filename(fileNum)
    end = "1948_2014.csv"

    fileName = base + middle + end
    data = read_csv_file(fileName)
    at = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            at.append(float(col) - 273.15)

    at = normalise_list(at)
    return at


def read_slp(fileNum):

    base = "..Data/SLP/daily_slp"
    middle = filename(fileNum)
    end = "1948_2014.csv"

    fileName = base + middle + end
    data = read_csv_file(fileName)
    slp = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            at.append(float(col)/10000)

    #at = normalise_list(at)
    return slp



def split_data(input1 , input2 , input3 , input4):

    """
    Splits a sequence into windows
    """
    seq =[i for i in range(INPUTS)]
    print(seq[0])

    seq[0] =  [np.array(input1[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input1) // INPUT_SIZE)]

    #seq[0] = normalise_list(seq[0])
    j = 1
    for inputs in input2:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]

        #seq[j] = normalise_list(seq[j])
        j+=1

    for inputs in input3:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]


        #seq[j] = normalise_list(seq[j])
        j+=1


    for inputs in input4:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]

        #seq[j] = normalise_list(seq[j])

        j+=1




    """
    print(seq1[0:10])
    print(seq2[0:10])
    print(seq3[0:10])
    print(seq4[0:10])
    """

    sequence = []
    for i in range(len(seq[0])):
        temp=[]
        for k in range(INPUTS):
            temp.append(seq[k][i])

        sequence.append(temp)

    #print(seq[0:10])

    X=[]
    y=[]
    y_org=[]


    for i in range(len(sequence) - NUM_STEPS - LEAD_TIME):

        z = []
        temp = np.array(sequence[i: i + NUM_STEPS+LEAD_TIME])
        #temp1 = normalize_seq(temp)
        X.append(temp[0:NUM_STEPS])
        #print("Y" , y)
    X = np.asarray(X , dtype=np.float32)


    return X



def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    test_size = int(len(X) * (1.0 - TRAIN_TEST_RATIO))

    X_train, X_test = X[:test_size], X[test_size:]

    y_train, y_test  = y[:test_size], y[test_size:]

    validation_size = int(len(X_train) * (1- TRAIN_VALIDATION_RATIO))
    X_train , X_validation = X_train[:validation_size] , X_train[validation_size:]
    y_train, y_validation  = y_train[:validation_size], y_train[validation_size:]

    return X_train , y_train , X_validation , y_validation , X_test , y_test


def process():
    rainfall = read_rainfall()

    uwindCI = read_uwind(1)
    uwindBOB = read_uwind(2)
    uwindSI = read_uwind(3)
    uwindAS = read_uwind(4)

    uwind = [uwindCI , uwindBOB , uwindAS , uwindSI]

    vwindCI = read_vwind(1)
    vwindSI = read_vwind(2)
    vwindAS = read_vwind(3)
    vwindBOB = read_vwind(4)

    vwind = [vwindCI , vwindBOB , vwindAS , vwindSI]

    atCI = read_at(1)
    atSI = read_at(2)
    atAS = read_at(3)
    atBOB = read_at(4)


    at = [atCI , atBOB , atAS , atSI]

    """
    print(len(slp))
    print(len(uwind))
    print(len(vwind))
    """

    X = split_data(rainfall,uwind , vwind , at )

    y = read_rainfall_class()
    y = y[NUM_STEPS+LEAD_TIME:]

    #print(X[0])
    #print(y[0])

    #print(X[1])
    #print(y[1])

    y = np.reshape(y , [len(y) , 3])
    X = np.reshape(X , [X.shape[0] , X.shape[1] , INPUTS])

    print(X.shape)
    print(y.shape)

    #print(X[0])
    #print(X[1])
    X_train , y_train , X_validation , y_validation , X_test , y_test = train_test_split(X,y)

    print(X_train.shape)
    print(y_train.shape)
    return X_train , y_train , X_validation , y_validation , X_test , y_test




if __name__ == '__main__':
    process()

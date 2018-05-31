import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn import preprocessing
import csv

import math

INPUT_SIZE = 1
NUM_STEPS =15#DAYS USED TO MAKE PREDICTION
LEAD_TIME = 5# PREDICITNG LEAD_TIME DAYS AHEAD
TRAIN_TEST_RATIO = 0.03
TRAIN_VALIDATION_RATIO = 0.03
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


def normalise_list(raw , size):

    mean = np.mean(raw[:size])
    std = np.std(raw[:size])

    norm = [(x - mean)/std for x in raw]
    return norm


def smooth_rainfall(train_data , size):
    EMA = 0.0
    gamma = 0.1

    for ti in range(size):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    return train_data
def one_hot_encode(x):

    if(x==0):
        return [1 , 0 , 0]

    if(x==1):
        return [0 , 1 , 0]

    if(x==2):
        return [0 , 0 , 1]

def read_rainfall():
    data = read_csv_file('../Data/Rainfall/daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append(float(col))


    l = len(rainfall) - NUM_STEPS - LEAD_TIME
    size = int(int(l*(1 - TRAIN_TEST_RATIO))*(1-TRAIN_VALIDATION_RATIO))
    print("size" ,size)

    #rainfall = normalise_list(rainfall , size)
    #rainfall = smooth_rainfall(rainfall , size)

    return rainfall , size


def read_rainfall_class():
    data = read_csv_file('../Data/Rainfall/class_daily_rainfall_central_India_1948_2014.csv')
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


def read_uwind(fileNum , size):

    base = '../Data/Uwind/daily_uwnd'
    middle = filename(fileNum)
    end = '1948_2014.csv'

    fileName = base + middle + end

    data = read_csv_file(fileName)
    uwnd = []

    """Creating list of Uwind data"""
    for row in (data):
        for col in row:

            uwnd.append(float(col))

    uwnd = normalise_list(uwnd , size)
    return uwnd


def read_vwind(fileNum , size):

    base = '../Data/Vwind/daily_vwnd'
    middle = filename(fileNum)
    end = '1948_2014.csv'

    fileName = base + middle + end
    data = read_csv_file(fileName)
    vwnd = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            vwnd.append(float(col))

    vwnd = normalise_list(vwnd , size)
    return vwnd


def read_at(fileNum , size):

    base = "../Data/AT/daily_at"
    middle = filename(fileNum)
    end = "1948_2014.csv"

    fileName = base + middle + end
    data = read_csv_file(fileName)
    at = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            at.append(float(col))

    at = normalise_list(at ,size)
    return at





def split_data(input1 , input2 , input3 , input4 , output):

    """
    Splits a sequence into windows
    """
    seq =[i for i in range(INPUTS)]

    seq[0] =  [np.array(input1[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
       for i in range(len(input1) // INPUT_SIZE)]

    j = 1
    for inputs in input2:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]

        j+=1

    for inputs in input3:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]

        j+=1


    for inputs in input4:

        seq[j] = [np.array(inputs[i * INPUT_SIZE: (i + 1) * INPUT_SIZE])
           for i in range(len(inputs) // INPUT_SIZE)]

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
        temp1 = np.array(sequence[i: i + NUM_STEPS+LEAD_TIME])
        X.append(temp1[0:NUM_STEPS])

        for j in range(LEAD_TIME):
            z.append(output[i+NUM_STEPS+j])

        #print("z" , z)
        y.append(z)
        #print("Y" , y)
    X = np.asarray(X , dtype=np.float32)
    y = np.asarray(y , dtype=np.float32)

    return X , y



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

    return X_train , y_train , X_validation , y_validation , X_test , y_test , test_size , validation_size


def process():
    rainfall, size = read_rainfall()

    plt.plot(rainfall)
    plt.show()

    uwindCI = read_uwind(1,size)
    uwindBOB = read_uwind(2,size)
    uwindSI = read_uwind(3,size)
    uwindAS = read_uwind(4,size)

    uwind = [uwindCI , uwindBOB , uwindAS , uwindSI]

    vwindCI = read_vwind(1,size)
    vwindSI = read_vwind(2,size)
    vwindAS = read_vwind(3,size)
    vwindBOB = read_vwind(4,size)

    vwind = [vwindCI , vwindBOB , vwindAS , vwindSI]

    atCI = read_at(1,size)
    atSI = read_at(2,size)
    atAS = read_at(3,size)
    atBOB = read_at(4,size)

    at = [atCI , atBOB , atAS , atSI]
    """
    print(len(slp))
    print(len(uwind))
    print(len(vwind))
    """

    rainfall_class = read_rainfall_class()

    X,y = split_data(rainfall,uwind , vwind , at , rainfall_class)


    y = np.reshape(y , [y.shape[0] , LEAD_TIME , 3])
    X = np.reshape(X , [X.shape[0] , X.shape[1] , INPUTS])

    print(X.shape)
    print(y.shape)

    """
    print(X[0])
    print(X[1])
    print(y[0])
    print(y[1])
    """

    X_train , y_train , X_validation , y_validation , X_test , y_test , test_size , validation_size = train_test_split(X,y)


    print(X_train.shape)
    print(y_train.shape)


    return X_train , y_train , X_validation , y_validation , X_test , y_test , rainfall , test_size , validation_size




if __name__ == '__main__':
    process()

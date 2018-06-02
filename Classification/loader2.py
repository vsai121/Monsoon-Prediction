import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import csv

import math

#Parameters for splitting data


NUM_STEPS =3  #DAYS USED TO MAKE PREDICTION
LEAD_TIME = 1  # PREDICITNG LEAD_TIME DAYS AHEAD

TRAIN_TEST_RATIO = 0.05
TRAIN_VALIDATION_RATIO = 0.05

INPUTS = 17 #Number of Variables used



def read_csv_file(filename):

    """
    Reading data from file filename

    """
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

    """

    Normalising input variables
    X = (X - mean(X_train)) / stddev(X_train)

    """
    mean = np.mean(raw[:size])
    std = np.std(raw[:size])

    norm = [(x - mean)/std for x in raw]
    return norm


def smooth_rainfall(train_data , size):
    EMA = 0.0
    gamma = 0.5

    for ti in range(size):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    return train_data


def one_hot_encode(x):

    """
    To one hot encode the three classes of rainfall

    """
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

    """
    To remove cases which have only 1 day of different class

    Example - 1 2 1 or 0 1 0

    """

    l = len(rainfall) - NUM_STEPS - LEAD_TIME
    size = int(int(l*(1 - TRAIN_TEST_RATIO))*(1-TRAIN_VALIDATION_RATIO))
    print("size" ,size)  #Training set size

    rainfall = normalise_list(rainfall , size)
    #rainfall = smooth_rainfall(rainfall , size)

    return rainfall , size


def read_rainfall_class():

    """
        Reading class of rainfall
    """
    data = read_csv_file('../Data/Rainfall/class4_daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append((int(col)))

    for i in range(1,len(rainfall)-1):

        if(rainfall[i-1]==1 and rainfall[i+1]==1):
            rainfall[i]=1

        if(rainfall[i-1]==2 and rainfall[i+1]==2):
            rainfall[i]=2

        if(rainfall[i-1]==0 and rainfall[i+1]==0):
            rainfall[i]=0


    for i in range(len(rainfall)):
        rainfall[i] = one_hot_encode(rainfall[i])

    #rainfall = normalise_list(rainfall)
    return rainfall



def filename(f):
    """
        file names of different region variables

        BOB - Bay of Bengal
        AS - Arabian sea

    """
    if(f==1):
        return '_central_India_'

    if (f==2):
        return '_south_India_'

    if(f==3):
        return '_BOB_'

    if(f==4):
        return '_AS_'


def read_uwind(fileNum , size):

    """
    Reading uwind of different regions
    """
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

    """
        Reading Vwind data
    """
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

    """
        Reading Air temperature data
    """

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



def read_pres(fileNum,size):

    base = "../Data/SLP/daily_pres"
    middle = filename(fileNum)
    end = "1948_2014.csv"

    fileName = base + middle + end
    data = read_csv_file(fileName)
    pres = []

    """Creating list of Vwind data"""
    for row in (data):
        for col in row:

            pres.append(float(col))

    pres = normalise_list(pres ,size)
    return pres

def split_data(input1 , input2 , input3 , input4 , input5 , output):

    """
    Splits input data  into windows

    """


    seq =[i for i in range(INPUTS)]

    #Appending all rainfall data in seq[0]
    seq[0] =  [np.array(input1[i * 1: (i + 1) * 1])
       for i in range(len(input1) // 1)]

    j = 1
    for inputs in input2:

        seq[j] = [np.array(inputs[i * 1: (i + 1) * 1])
           for i in range(len(inputs) // 1)]

        j+=1

    for inputs in input3:

        seq[j] = [np.array(inputs[i * 1: (i + 1) * 1])
           for i in range(len(inputs) // 1)]

        j+=1


    for inputs in input4:

        seq[j] = [np.array(inputs[i * 1: (i + 1) * 1])
           for i in range(len(inputs) // 1)]

        j+=1

    for inputs in input5:

        seq[j] = [np.array(inputs[i * 1: (i + 1) * 1])
           for i in range(len(inputs) // 1)]

        j+=1


    """
    print(seq1[0:10])
    print(seq2[0:10])
    print(seq3[0:10])
    print(seq4[0:10])

    """

    sequence = []
    """

    Concantenating values of different variables into a list

    seq = [rainfall , uwind(4) , vwind(4) , air_temperature(4) , pressure(4)]

    (4) - for the 4 regions

    """
    for i in range(len(seq[0])):
        temp=[]
        for k in range(INPUTS):
            temp.append(seq[k][i])

        sequence.append(temp)

    #print(seq[0:10])

    X=[]
    y=[]

    """
    X = [[Day1 variables] , [Day 2 variables]..... [DayNUM_STEPS variables]]
    y = [DayNUM_STEPS+LEAD_TIME+ rainfall class]

    """
    for i in range(len(sequence) - NUM_STEPS - LEAD_TIME):

        z = []
        temp1 = np.array(sequence[i: i + NUM_STEPS])
        X.append(temp1[0:NUM_STEPS])

        z.append(output[i+NUM_STEPS+LEAD_TIME-1])

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

    """
    Reading variables

    """
    rainfall, size = read_rainfall()

    #plt.plot(rainfall)
    #plt.show()

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

    presCI = read_pres(1,size)
    presSI = read_pres(2,size)
    presAS = read_pres(3,size)
    presBOB = read_pres(4,size)


    pres = [presCI , presBOB , presAS , presSI]

    rainfall_class = read_rainfall_class()

    X,y = split_data(rainfall,uwind , vwind , at , pres , rainfall_class)


    y = np.reshape(y , [y.shape[0] , 3])
    X = np.reshape(X , [X.shape[0] , X.shape[1] , INPUTS])

    print(X.shape)
    print(y.shape)


    """
    Checking if X and y have been split correctly

    print(X[0])
    print(X[1])
    print(y[0])
    print(y[1])


    """

    X_train , y_train , X_validation , y_validation , X_test , y_test , test_size , validation_size = train_test_split(X,y)


    print(X_train.shape)
    print(y_train.shape)

    """
    Data split into train validation and test

    """

    return X_train , y_train , X_validation , y_validation , X_test , y_test



if __name__ == '__main__':
    process()

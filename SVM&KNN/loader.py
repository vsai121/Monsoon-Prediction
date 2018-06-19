import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import csv

import math

import random



#Parameters for splitting data


NUM_STEPS = 20 #DAYS USED TO MAKE PREDICTION
LEAD_TIME = 5# PREDICITNG LEAD_TIME DAYS AHEAD

TRAIN_TEST_RATIO = 0.1

INPUTS= 3 #Number of Variables used

freq = 1./4  # Hours
window_size = 25
pad = np.zeros(window_size) * np.NaN

def lanc(numwt, haf):

    summ = 0
    numwt += 1
    wt = np.zeros(numwt)

    # Filter weights.
    ii = np.arange(numwt)
    wt = 0.5 * (1.0 + np.cos(np.pi * ii * 1. / numwt))
    ii = np.arange(1, numwt)
    xx = np.pi * 2 * haf * ii
    wt[1:numwt + 1] = wt[1:numwt + 1] * np.sin(xx) / xx
    summ = wt[1:numwt + 1].sum()
    xx = wt.sum() + summ
    wt /= xx
    return np.r_[wt[::-1], wt[1:numwt + 1]]



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
    data = read_csv_file('../Data/Rainfall/normalized_daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    months=[]

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:

            rainfall.append(float(col))



    for row in (data):
        days = 0
        for col in row:

            days+=1
            if(days<=30):
                months.append(0)

            elif(days<=61):
                months.append(1)

            elif(days<=92):
                months.append(2)

            else:
                months.append(3)


    l = len(rainfall) - NUM_STEPS - LEAD_TIME
    #print("l",l)
    size = l - int(l*TRAIN_TEST_RATIO)
    #print("size" ,size)  #Training set size
    rainfall = normalise_list(rainfall,size)

    wt = lanc(window_size, freq)
    rainfall = np.convolve(wt, rainfall, mode='same')

    return rainfall , months , size


def read_rainfall_class():

    """
        Reading class of rainfall
    """
    ratio = [0.,0.,0.]
    data = read_csv_file('../Data/Rainfall/class4_daily_rainfall_central_India_1948_2014.csv')
    rainfall = []

    """Creating list of rainfall data"""
    for row in (data):
        for col in row:
            rainfall.append((int(col)))

    for i in range(1,len(rainfall)-1):


        if(rainfall[i-1]==2 and rainfall[i+1]==2):
            rainfall[i]=2

        elif(rainfall[i-1]==0 and rainfall[i+1]==0):
            rainfall[i]=0

    for i in range(1,len(rainfall)-1):
        if(rainfall[i-1]==1 and rainfall[i+1]==1):
            rainfall[i]=1

    zeros=0
    ones=0
    twos=0
    for i in range(len(rainfall)):
        if(rainfall[i]==0):
            zeros+=1

        if(rainfall[i]==1):
            ones+=1

        if(rainfall[i]==2):
            twos+=1


    #rainfall = normalise_list(rainfall)
    print("zeros",zeros)
    print("twos",twos)
    total = zeros + ones + twos
    ratio[0] = float(ones+twos)/total
    ratio[1] = float(zeros+twos)/total
    ratio[2] = float(zeros+ones)/total

    return rainfall,ratio


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

    base = "../Data/SLP/daily_slp"
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

def split_data(input1 , input2 , input3 , input4 , input5 , output , size):

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
    new_size=0
    print("seq",len(sequence))
    for i in range(len(sequence) - NUM_STEPS - LEAD_TIME):

        temp1 = np.array(sequence[i: i + NUM_STEPS])

        new_size += 1

        X.append(temp1[0:NUM_STEPS])
        z=[]

        y.append(output[i+NUM_STEPS+LEAD_TIME-1])


    X = np.asarray(X , dtype=np.float32)
    y = np.asarray(y , dtype=np.float32)

    return X , y , new_size



def train_test_split(X , y):

    """
    Splitting data into training and test data"
    """

    size = X.shape[0]
    test_size = int(size * TRAIN_TEST_RATIO)

    X_test = X[-test_size:]
    y_test = y[-test_size:]

    X_train = X[:-test_size]
    y_train =  y[:-test_size]

    return X_train , y_train , X_test , y_test , test_size


def process():

    """
    Reading variables

    """
    rainfall, month , size = read_rainfall()

    #plt.plot(rainfall)
    #plt.show()

    uwindCI = read_uwind(1,size)
    uwindBOB = read_uwind(2,size)
    uwindSI = read_uwind(3,size)
    uwindAS = read_uwind(4,size)

    uwind = []

    vwindCI = read_vwind(1,size)
    vwindSI = read_vwind(2,size)
    vwindAS = read_vwind(3,size)
    vwindBOB = read_vwind(4,size)

    vwind = []

    atCI = read_at(1,size)
    atSI = read_at(2,size)
    atAS = read_at(3,size)
    atBOB = read_at(4,size)

    at = [atCI,]

    presCI = read_pres(1,size)
    presSI = read_pres(2,size)
    presAS = read_pres(3,size)
    presBOB = read_pres(4,size)

    pres=[presCI,]

    months = []
    rainfall_class , ratio  = read_rainfall_class()

    X,y,new_size = split_data(rainfall,pres , uwind , at ,vwind, rainfall_class , size)


    y = np.reshape(y , [y.shape[0] , 1 ])
    X = np.reshape(X , [X.shape[0] , X.shape[1]*INPUTS])

    print(X.shape)
    print(y.shape)

    X_train , y_train , X_test , y_test , test_size  = train_test_split(X,y)


    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)


    """
    Data split into train and test

    """

    return X_train , y_train , X_test , y_test , ratio



if __name__ == '__main__':
    process()

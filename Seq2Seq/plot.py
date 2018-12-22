import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import csv
import sys

def plot():

    arguments = sys.argv[1:]
    count = len(arguments)
    print("Count ",count)
    if(count > 1 or count == 0):
        return None

    filename = arguments[0]

    print(filename)

    updated_pred = []
    updated_act = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:

            print('Actual -' +  row[0] +  '\t  Predicted - ' + row[1])
            updated_act.append(int(row[0]))
            updated_pred.append(int(row[1]))
            line_count += 1
        print(f'Processed {line_count} lines.')

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)

    ax1 = fig.add_subplot(111)

    line1 = ax1.plot(updated_pred,'bo-',label='list 1')
    line2 = ax1.plot(updated_act,'go-',label='list 2')

    plt.show()

    print("Confusion matrix" , confusion_matrix(updated_act,updated_pred))

    print("F1 score" , f1_score(updated_act, updated_pred, average='micro'))

    active_spells = []
    dry_spells = []

    predicted_active_spells = []
    predicted_dry_spells = []


    i = 0
    while(i<len(updated_pred)):
        temp = []
        if(updated_act[i]==2):

            while(i<len(updated_pred) and updated_act[i]==2):

                temp.append(i)
                i+=1

            if(len(temp)>2):
                active_spells.append(temp)

        elif(updated_act[i]==0):


            while(i<len(updated_pred) and updated_act[i]==0):

                temp.append(i)
                i+=1

            if(len(temp)>2):
                dry_spells.append((temp))

        else:
            i+=1



    print("\n\n\n")

    i = 0


    predActive = 0
    predDry=0
    while(i<len(updated_pred)):
        temp = []
        if(updated_pred[i]==2):

            while(i<len(updated_pred) and updated_pred[i]==2):

                temp.append(i)
                i+=1

            if(len(temp)>0):
                predicted_active_spells.append(temp)

        elif(updated_pred[i]==0):


            while(i<len(updated_pred) and updated_pred[i]==0):


                temp.append(i)
                i+=1

            if(len(temp)>0):
                predicted_dry_spells.append(temp)

        else:
            i+=1

    print("active_spells" , active_spells)
    print("predicted active_spells" , predicted_active_spells)
    print("\n\n")

    print("Dry spells" , dry_spells)
    print("predicted Dry spells" , predicted_dry_spells)

    print(len(active_spells))
    print(len(predicted_active_spells))
    print(len(dry_spells))
    print(len(predicted_dry_spells))


    print(predActive)
    print(predDry)


if __name__ == '__main__':
    plot()

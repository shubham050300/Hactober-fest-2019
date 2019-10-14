import pandas
import numpy as np
import math
import random

def accuracy(y1, y2):
    if len(y1) != len(y2):
        return -1
    count = 0
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            count += 1
    return count / len(y1)

def recall(pred, y_test):
    tp=0
    fn=0
    for i in range(len(y_test)):
        if( y_test[i]==pred[i] and y_test[i]==1):
            tp+=1
        if( y_test[i]==0 and pred[i]==1):
            fn+=1
    recall_value=tp/(tp+fn)
    return recall_value

def precision(pred, y_test):
    tp=0
    fp=0
    for i in range(len(y_test)):
        if(y_test[i]==pred[i] and y_test[i]==1):
            tp+=1
        if(y_test[i]==1 and pred[i]==0):
            fp+=1
    precision_value=tp/(tp+fp)
    return precision_value

def load_data():
    data = pandas.read_csv('IRIS.csv')
    data = data.reindex(np.random.permutation(data.index))
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1] == 'Yes'
    return (x, y)

def perceptron_predict(t, data_x, data_y, x, y_test):
    learning_rate=0.05
    w = [0] * (data_x.shape[1])
    for j in range(len(w)): 
        w[j] = (random.randint(-1, 1))
 
    for i in range (500):   #no of epochs

        for j in range (data_x.shape[0]):
            sum=-0.5
            for k in range (data_x.shape[1]):
                sum+=w[k]*data_x[j][k]
            y=1 if sum >=0 else 0
            error=data_y[j]-y;
            for k in range (data_x.shape[1]):
                delta_w = learning_rate*error*data_x[j][k]
                w[k]+=delta_w

        res_y = [0] * len(x)
    
    for z in range(len(x)):
        sum=-0.5
        for j in range(x.shape[1]):
            sum+=w[j]*x[z][j]
        ans_y = 1 if sum >=0 else 0
        res_y[z]=ans_y

        # return res_y
        # res = accuracy(res_y, y_test)
        # print(f'fold-{t+1} epoch:{i+1}  acc: {res}')

    return res_y

def k_10_fold_perceptron(x, y):
    l = math.ceil(len(x)/10)
    res = [0] * 10
    prec = [0] * 10
    rec = [0] * 10
    for i in range(10):
        x_test = x.iloc[i*l:(i+1)*l].values
        y_test = y.iloc[i*l:(i+1)*l].values
        x_train = x.iloc[:i*l].append(x.iloc[(i+1)*l:], ignore_index=True).values
        y_train = y.iloc[:i*l].append(y.iloc[(i+1)*l:], ignore_index=True).values
        prediction = perceptron_predict(i, x_train, y_train, x_test, y_test)
        res[i] = accuracy(prediction, y_test)
        prec[i] = precision(prediction, y_test)
        rec[i] = recall(prediction, y_test)
        print(f'fold {i+1}:accuracy: {res[i]} precision: {prec[i]} recall: {rec[i]}')
    return res,prec,rec

def main():
    x, y = load_data()
    res, prec, rec = k_10_fold_perceptron(x, y)
    print(f'max accuracy: {max(res)}\navg: {sum(res)/len(res)}\navg prec: {sum(prec)/len(prec)}\navg recall: {sum(rec)/len(rec)}')

if __name__ == '__main__':
    main()
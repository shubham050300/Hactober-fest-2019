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
    data = pandas.read_csv('SPECT.csv')
    data = data.reindex(np.random.permutation(data.index))
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0] == 'Yes'
    return (x, y)

def neural_network_predict(t, x_train, y_train, x_test, y_test):
    l = len(x_train[0])
    hidden_l = 5
    learning_rate=0.2

    w1 = [[ random.uniform(-0.1, 0.1) for i in range(hidden_l)] for j in range(l)]
    w2 = [ random.uniform(-0.1,0.1) for i in range(hidden_l)]

    theta1=[random.uniform(-1,1) for i in range(hidden_l)]
    theta2=random.uniform(-1,1) 

    ip = [0]*(hidden_l+1)
    op = [0]*(hidden_l+1)

    for i in range (500):   #no of epochs
        for j in range(x_train.shape[0]):
            for k in range (hidden_l):
                sumv=0
                for m in range(x_train.shape[1]):
                    sumv+= (w1[m][k])*x_train[j][m]
                ip[k]=sumv+theta1[k]

            for k in range(hidden_l):
                op[k]=1/(1+math.exp(-1*ip[k]))


            sum1=0
            for k in range(hidden_l):
                sum1+=(w2[k]*op[k])

            ip[hidden_l]=sum1+theta2
            op[hidden_l]=1/(1+math.exp(-1*ip[hidden_l]))

            #backpropagation
            err_op=op[hidden_l]*(1-op[hidden_l])*(y_train[j]-op[hidden_l])
            err_hidden=[0]*hidden_l
            for k in range(hidden_l):
                err_hidden[k]=op[k]*(1-op[k])*err_op*w2[k]

            for k in range(hidden_l):
                w2[k]+=learning_rate*err_op*op[k]

            for k in range(x_train.shape[1]):
                for m in range(hidden_l):
                    w1[k][m]+=learning_rate*err_hidden[m]*x_train[j][k]

            for k in range(hidden_l):
                theta1[k]+=learning_rate*err_hidden[k]

            theta2+=learning_rate*err_op

    res_y=[0]* len(x_test)

    pred_in=[0]*(hidden_l+1)
    pred_out=[0]*(hidden_l+1)
    for j in range(len(x_test)):
        for k in range(hidden_l):
            sumv=0
            for m in range(x_test.shape[1]):
                sumv+=(w1[m][k]*x_test[j][m])
            pred_in[k]=sumv+theta1[k]
        for k in range(hidden_l):
            pred_out[k]=1/(1+math.exp(-1*pred_in[k]))

        sum1=0
        for k in range(hidden_l):
            sum1+=(w2[k]*pred_out[k])

        pred_in[hidden_l]=sum1+theta2
        pred_out[hidden_l]=1/(1+math.exp(-1*pred_in[hidden_l]))
    

        res_y[j]=1 if pred_out[hidden_l]>0.5 else 0
        print( pred_out[hidden_l],"&&&",res_y[j])

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
        prediction = neural_network_predict(i, x_train, y_train, x_test, y_test)
        res[i] = accuracy(prediction, y_test)
        prec[i] = precision(prediction, y_test)
        rec[i] = recall(prediction, y_test)
        print(f'fold {i+1}:accuracy: {res[i]} precision: {prec[i]} recall: {rec[i]}')
    return res,prec,rec

def main():
    x, y = load_data()
    res, prec, rec = k_10_fold_perceptron(x, y)
    print('\n')
    print(f'max accuracy: {max(res)}\navg accuracy: {sum(res)/len(res)}\nprecision: {sum(prec)/len(prec)}\nrecall: {sum(rec)/len(rec)}')

if __name__ == '__main__':
    main()
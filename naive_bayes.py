import pandas
import numpy as np

def accuracy(y1, y2):
    if len(y1) != len(y2):
        return -1
    count = 0
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            count += 1
    return count / len(y1)

def load_data():
    data = pandas.read_csv('SPECT.csv')
    data = data.reindex(np.random.permutation(data.index))
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0] == 'Yes'
    return (x, y)

def naive_bayes_predict(data_x, data_y, x):
    probabilities = [([0] * data_x.shape[1]).copy() for i in range(2)]
    p_count = 0
    n_count = 0
    for i in range(len(data_x)):
        if data_y[i] == 1:
            p_count += 1
        else:
            n_count += 1
            
        for j in range(data_x.shape[1]):
            if data_y[i] == 1 and data_x[i][j] == 1:
                probabilities[1][j] += 1
            if data_y[i] == 0 and data_x[i][j] == 1:
                probabilities[0][j] += 1
    for i in range(data_x.shape[1]):
        probabilities[1][i] /= p_count
        probabilities[0][i] /= n_count
    
    res_y = [0] * len(x)
    
    for i in range(len(x)):
        p_yes = p_count / len(data_x)
        p_no = n_count / len(data_x)
        for j in range(data_x.shape[1]):
            p_yes *= probabilities[1][j] if x[i][j] == 1 else 1 - probabilities[1][j]
            p_no *= probabilities[0][j] if x[i][j] == 1 else 1 - probabilities[0][j]
        if p_yes > p_no:
            res_y[i] = 1

    return res_y

def k_10_fold_nb(x, y):
    l = len(x) // 10
    res = [0] * 10
    for i in range(10):
        x_test = x.iloc[i*l:(i+1)*l].values
        y_test = y.iloc[i*l:(i+1)*l].values
        x_train = x.iloc[:i*l].append(x.iloc[(i+1)*l:], ignore_index=True).values
        y_train = y.iloc[:i*l].append(y.iloc[(i+1)*l:], ignore_index=True).values
        prediction = naive_bayes_predict(x_train, y_train, x_test)
        res[i] = accuracy(prediction, y_test)
        print(f'fold {i+1}:accuracy: {res[i]}\n')
    return res

def main():
    x, y = load_data()
    res = k_10_fold_nb(x, y)
    print(f'max accuracy: {max(res)}\navg: {sum(res)/len(res)}')

if __name__ == '__main__':
    main()
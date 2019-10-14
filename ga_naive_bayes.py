import numpy as np
import pandas as pd
import random

def load_dataset():
    data = pd.read_csv('SPECT.csv')
    data = data.reindex(np.random.permutation(data.index))
    a = data.iloc[:, 1:]
    b = data.iloc[:, 0] == 'Yes'
    return (a, b)

def naive_bayes_algo(data_x, data_y, test_x):

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
    
    res_y = [0] * len(test_x)
    
    for i in range(len(test_x)):
        p_yes = p_count / len(data_x)
        p_no = n_count / len(data_x)
        for j in range(data_x.shape[1]):
            if test_x[i][j] == 1:
                p_yes *= probabilities[1][j]
            else:
                p_yes*= (1 - probabilities[1][j])

            if test_x[i][j] == 1:
                p_no *= probabilities[0][j]
            else:
                p_no*=(1 - probabilities[0][j])
        if p_yes > p_no:
            res_y[i] = 1

    return res_y

def accuracy(y1, y2):
    count = 0
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            count += 1
    ans = count / len(y1)
    return ans

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

def cross_fold(x, y):
    l = len(x) // 10
    res = [0] * 10
    prec = [0] * 10
    rec = [0] * 10
    for i in range(10):
        x_test = x.iloc[i*l:(i+1)*l].values
        y_test = y.iloc[i*l:(i+1)*l].values
        x_train = x.iloc[:i*l].append(x.iloc[(i+1)*l:], ignore_index=True).values
        y_train = y.iloc[:i*l].append(y.iloc[(i+1)*l:], ignore_index=True).values
        prediction = naive_bayes_algo(x_train, y_train, x_test)
        res[i] = accuracy(prediction, y_test)
        prec[i] = precision(prediction, y_test)
        rec[i] = recall(prediction, y_test)
        # print('cross-fold {i+1} -> accuracy: {res[i]}')
    return res,prec,rec

#---------------------------------------------------------------------------------


def evaluate_fitness(x,y,pop):

    fitness_acc=[]
    for i in range(len(pop)):
        col=[]
        for j in range(len(pop[0])):
            if(pop[i][j]==1):
                col.append(j)
        data_x = x.iloc[:,col]
        res, prec, rec=cross_fold(data_x,y)
        fitness_acc.append(sum(res)/len(res))
    # print("DONE")
    return fitness_acc


def roulette(fitness,pop, m):
    chrom=[0]*m
    total_prob=0
    # for i in range(m):
    #     print(fitness[i]) 
    # print("&&&")
    for i in range(m):
        total_prob=total_prob+fitness[i]
    prob=[0]*m
    for i in range(m):
        prob[i]=fitness[i]/total_prob
    cumm_prob=[0]*m
    cumm_prob[0]=prob[0]
    for i in range(1,m):
        cumm_prob[i]=cumm_prob[i-1]+prob[i]
    # print(comm_prob[n-1],"&&&")
    random_no=[random.random() for i in range(m)]
    # for i in range(m):
    #     print(random_no[i])
    # print("^^^")
    for i in range(m):
        for j in range(m):
            if(random_no[i]<cumm_prob[j]):
                chrom[i]=j
                break
    # for i in range(m):
    #     print(chrom[i])
    # print("*****")
    new_pop=[[0 for i in range(len(pop[0]))] for j in range(len(pop))]
    for i in range(m):
        new_pop[i]=pop[chrom[i]]

    # print("selection!!!!!")
    # for i in range(m):
    #     print(" ")
    #     for j in range(22):
    #         print(new_pop[i][j], end='')
    return new_pop

def crossover(pop,n,m):
    t=int(0.25*m)
    for i in range(t):
        a=random.randint(0, m-1)
        b=random.randint(0, m-1)
        crossover_len=random.randint(1, n-1)
        for j in range(crossover_len,n):
            pop[a][j]=pop[b][j]
    # print("crosssover!!!!!")
    # for i in range(m):
    #     print(" ")
    #     for j in range(n):
    #         print(pop[i][j], end='')
    return pop

def mutation(pop,n,m):
    t=int(0.1*m)
    for i in range(t):
        a=random.randint(0,m-1)
        b=random.randint(0,n-1)
        pop[a][b]=1-pop[a][b]
    # print("mutation!!!!!")
    # for i in range(m):
    #     print(" ")
    #     for j in range(n):
    #         print(pop[i][j], end='')

    return pop




def main():
    x,y = load_dataset()
    chromosome_len=x.shape[1]
    pop_size=30
    pop=[[random.randint(0, 1) for i in range(chromosome_len)] for j in range(pop_size)]

    for i in range(10):
        fitness=evaluate_fitness(x,y,pop)
        # print("DONEEE")
        after_roulette_selection=roulette(fitness,pop,pop_size)
        after_crossover=crossover(after_roulette_selection, chromosome_len, pop_size)
        after_mutation=mutation(after_crossover,chromosome_len,pop_size)
        pop=after_mutation
        print("Iteration",(i+1),"done!")

    # print("Optimal solution gene set is:")
    # for i in range(pop_size):
    #     print(" ")
    #     for j in range(chromosome_len):
    #         print(pop[i][j], end='')

    best_chromosome=-1
    max_acc=-1
    accuracy1=[0]*pop_size
    precision1=[0]*pop_size
    recall1=[0]*pop_size
    for i in range(len(pop)):
        col=[]
        for j in range(len(pop[0])):
            if(pop[i][j]==1):
                col.append(j)
        data_x = x.iloc[:,col]
        res, prec, rec=cross_fold(data_x,y)
        accuracy1[i]=sum(res)/len(res)
        precision1[i]=sum(prec)/len(prec)
        recall1[i]=sum(rec)/len(rec)
        if(accuracy1[i]>max_acc):
            max_acc=accuracy1[i]
            best_chromosome=i

    print("selected chromosome:")
    for i in range(chromosome_len):
        print(pop[best_chromosome][i],end='')

    print("\n")
    print("accuracy:",accuracy1[best_chromosome])
    print("precision:",precision1[best_chromosome])
    print("recall:",recall1[best_chromosome])

        










if __name__ == '__main__':
    main()
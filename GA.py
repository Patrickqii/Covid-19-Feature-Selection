from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import random

data = pd.read_excel("/Users/patrickqi/Desktop/毕设/Dataset.xlsx")
data1 = data.iloc[:, 0:8]
data2 = np.mat(data1)
tezheng = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']

pc = 0.02
t = 200
n = 30#种群中个体数目

def jqSubset(tezheng, xulie):
    tz = []
    for num in range(0, len(xulie)):
        if xulie[num] == 1:
            tz.append(tezheng[num])
    return tz
#遗传算法主体
def GA(d):
    population = np.zeros((n, 8))
    for i in range(n):
        a = np.zeros(8-d)
        b = np.ones(d)
        c = np.append(a, b)
        c = (np.random.permutation(c.T)).T#打乱顺序
        population[i] = c

    for i in range(t):
        fitness = np.zeros(n)
        for j in range(n):
            fitness[j] = Cv(population[j])
        population = selection(population, fitness)#根据轮盘赌选择
        population = crossover(population)#交叉产生新个体
        population = mutation(population)#有概率变异产生新个体

    best_fitness = max(fitness)
    best_people = population[fitness.argmax()]

    return best_people, best_fitness, population

#种群选择（轮盘赌）
def selection(population, fitness):
    fitness_sum = np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i-1]
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    population_new = np.zeros((n, 8))
    for i in range(n):
        rand = np.random.uniform(0, 1)
        for j in range(n):
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] = population[j]
            else:
                if fitness_sum[j-1] < rand and rand <= fitness_sum[j]:
                    population_new[i]=population[j]
    return population_new

def crossover(population):
    father = population[0:7, :]
    mother = population[7:, :]
    np.random.shuffle(father)#将顺序打乱
    np.random.shuffle(mother)
    #随机选择两个交叉点，从而选出进行交叉的片段（这里我们不考虑所选特征数目的变化）
    for i in range(7):
        '''
        cross_point_1 = random.randint(0, 6)
        cross_point_2 = random.randint(cross_point_1 + 1, 7)
        tmp = father[i, cross_point_1:cross_point_2]
        father[i, cross_point_1:cross_point_2] = mother[i, cross_point_1:cross_point_2]
        mother[i, cross_point_1:cross_point_2] = tmp
        '''
        father_1 = father[i]
        mother_1 = mother[i]
        one_zero = []
        zero_one = []
        for j in range(8):
            if father_1[j] == 1 and mother_1[j] == 0:
                one_zero.append(j)
            if father_1[j] == 0 and mother_1[j] == 1:
                zero_one.append(j)
        length1 = len(one_zero)
        length2 = len(zero_one)
        length = max(length1, length2)
        half_length = int(length / 2)  # half_length为交叉的位数
        for k in range(half_length):  # 进行交叉操作
            p = one_zero[k]
            q = zero_one[k]
            father_1[p] = 0
            mother_1[p] = 1
            father_1[q] = 1
            mother_1[q] = 0
        father[i] = father_1  # 将交叉后的个体替换原来的个体
        mother[i] = mother_1
    population = np.append(father, mother, axis=0)
    return population

def mutation(population):
    for i in range(n):
        c = np.random.uniform(0, 1)
        if c <= pc:
            mutation_s = population[i]
            zero = []  # zero存的是变异个体中第几个数为0
            one = []  # one存的是变异个体中第几个数为1
            for j in range(8):
                if mutation_s[j] == 0:
                    zero.append(j)
                else:
                    one.append(j)

            if (len(zero) != 0) and (len(one) != 0):
                a = np.random.randint(0, len(zero))  # e是随机选择由0变为1的位置
                b = np.random.randint(0, len(one))  # f是随机选择由1变为0的位置
                e = zero[a]
                f = one[b]
                mutation_s[e] = 1
                mutation_s[f] = 0
                population[i] = mutation_s

    return population

def Cv(subset):
    tz = jqSubset(tezheng, subset)
    x = data[tz]
    y = data['Target']
    # clf = svm.SVC(kernel='linear')
    # clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=10)#基分类器数目为10
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    cv_results = cross_validate(clf, x, y, scoring='accuracy')
    accuracy = float(cv_results['test_score'].mean())
    return accuracy

if __name__ =='__main__':
    best_people, best_fitness, population = GA(5)
    tz = jqSubset(tezheng, best_people)
    x = data[tz]
    y = data['Target']
    #clf = svm.SVC(kernel='linear')
    #clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=10)#基分类器数目为10
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    score_f = cross_val_score(clf, x, y, cv=5, scoring='f1').mean()
    score_precision = cross_val_score(clf, x, y, cv=5, scoring='precision').mean()
    score_recall = cross_val_score(clf, x, y, cv=5, scoring='recall').mean()
    score_roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc').mean()
    print("特征子集：")
    print(best_people)
    print("accuracy:%f" % best_fitness)
    print("F1-Score:%f" % score_f)
    print("Precision:%f" % score_precision)
    print("recall:%f" % score_recall)
    print("AUC值：%f" % score_roc_auc)
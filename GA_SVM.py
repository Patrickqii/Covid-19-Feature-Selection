from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import random
import math
from minepy import MINE

data = pd.read_excel("/Users/patrickqi/Desktop/毕设/Dataset.xlsx")
data1 = data.iloc[:, 0:8]
data2 = np.mat(data1)
tezheng = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']

p_cross = 0.7       #交叉概率
p_mutation = 0.1    #变异概率
t = 100     #遗传算法迭代次数
n = 40     #种群个体数
v = 0.8
weight_mic = 2/3
weight_rich = 1/6
weight_corr = 1/6

def jqSubset(tezheng, xulie):
    tz = []
    for num in range(0, len(xulie)):
        if xulie[num]==1:
            tz.append(tezheng[num])
    return tz

#遗传算法主体
def GA_SVM():
    population = np.zeros((n, 8))
    #随机初始化种群
    for i in range(n):
        for j in range(8):
            k = np.random.uniform(0, 1)
            if k >= 0.5:
                population[i, j] = 1
    #计算初始种群的适应度
    fitness = np.zeros(n)
    for i in range(n):
        subset = population[i]
        fitness[i] = Ind_score(subset, population, 0)
    for i in range(t):
        population = selection(population, fitness)
        population = crossover(population)
        population = mutation(population)
        for j in range(n):
            subset = population[j]
            fitness[j] = Ind_score(subset, population, i)

    best_fitness = max(fitness)
    best_people = population[fitness.argmax()]

    return best_people, best_fitness, population

#选择操作
def selection(population, fitness):
    fitness_sum = np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i - 1]
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    population_new = np.zeros((n, 8))
    argsort_fitness = np.argsort(fitness)
    e = argsort_fitness[::-1]
    #保留fitness前10%的个体
    for i in range(int(n/10)):
        population_new[i] = population[e[i]]
    for i in range(int(n/10), n):
        rand = np.random.uniform(0, 1)
        for j in range(n):
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] = population[j]
                    break
            else:
                if fitness_sum[j-1] < rand and rand <= fitness_sum[j]:
                    population_new[i] = population[j]
                    break
    return population_new

#交叉操作
def crossover(population):
    father = population[0:int(n/2), :]
    mother = population[int(n/2):n, :]
    np.random.shuffle(father)  # 将父代个体按行打乱以随机配对
    np.random.shuffle(mother)
    for i in range(int(n/2)):
        rand = np.random.uniform(0, 1)
        if rand <= p_cross:
            cross_point1 = random.randint(0, 6)     #寻找两个随机交叉点
            cross_point2 = random.randint(cross_point1+1, 7)
            '''
            tmp = father[i, cross_point1:cross_point2+1]
            father[i, cross_point1:cross_point2+1] = mother[i, cross_point1:cross_point2+1]
            mother[i, cross_point1:cross_point2+1] = tmp
            '''
            for j in range(cross_point1, cross_point2+1):
                tmp = father[i, j]
                father[i, j] = mother[i, j]
                mother[i, j] = tmp
    population = np.append(father, mother, axis=0)
    return population

#变异操作
def mutation(population):
    for i in range(n):
        rand = np.random.uniform(0, 1)
        if rand <= p_mutation:
            rand_posi = random.randint(0, 7)
            if population[i, rand_posi] == 0:
                population[i, rand_posi] = 1
            else:
                population[i, rand_posi] = 0
    return population

#综合适应度函数
def Ind_score(subset, population, ii):
    weight_decat = pow(v, ii+1)
    acc_score = fit(subset)
    multi_score = MultiScore(subset, population)
    score = (1-weight_decat)*acc_score+weight_decat*multi_score
    return score

#求交叉验证准确率的函数
def fit(subset):
    tz = jqSubset(tezheng, subset)
    if len(tz) == 0:
        return 0
    x = data[tz]
    y = data['Target']
    #clf = svm.SVC(kernel='linear')
    #clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=10)  # 基分类器数目为10
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    cv_results = cross_validate(clf, x, y, scoring='accuracy')
    accuracy = float(cv_results['test_score'].mean())
    return accuracy

def pjzhunze(subset):
    tz = jqSubset(tezheng, subset)
    if len(tz) == 0:
        return 0
    x = data[tz]
    y = data['Target']
    #clf = svm.SVC(kernel='linear')
    #clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=10)  # 基分类器数目为10
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    score_acc = cross_val_score(clf, x, y, cv=5, scoring='accuracy').mean()
    score_f = cross_val_score(clf, x, y, cv=5, scoring='f1').mean()
    score_precision = cross_val_score(clf, x, y, cv=5, scoring='precision').mean()
    score_recall = cross_val_score(clf, x, y, cv=5, scoring='recall').mean()
    score_roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc').mean()
    return score_acc, score_f,score_precision,score_recall, score_roc_auc

#个体与类别之间的相关性
def mic_score(subset):
    num = 0
    score = 0
    for i in range(8):
        if subset[i] == 1:
            num += 1
            xx = data.iloc[:, i]
            xx = xx.values
            yy = data['Target']
            yy = yy.values
            m = MINE()
            m.compute_score(xx, yy)
            score += m.mic()
    if num != 0:
        score /= num
    return score

#个体对种群特征基因丰富度的影响
def rich_score(subset, population):
    ind = np.zeros(8)
    for i in range(8):
        ss = 0
        for j in range(n):
            if population[j, i] == 1:
                ss += 1
        ind[i] = ss/n
    score = 0
    num = 0
    for i in range(8):
        if subset[i] == 1:
            score += ind[i]
            num += 1
    if num != 0:
        score /= num
    return score

#个体内部特征基因冗余的程度
def corr_score(subset):
    num = 0
    score = 0
    for i in range(8):
        if subset[i] == 1:
            num += 1
    for i in range(8):
        if subset[i] == 1:
            for j in range(8):
                if subset[j] == 1 and i != j:
                    xx = data.iloc[:, i]
                    xx = xx.values
                    yy = data.iloc[:, j]
                    yy = yy.values
                    corr_matrix = np.corrcoef(xx, yy)
                    score += abs(corr_matrix[0, 1])
    if num != 0:
        score /= num
    return score


#适应度函数中考虑种群影响的部分
def MultiScore(subset, population):
    score = weight_mic*mic_score(subset)-weight_rich*rich_score(subset, population)+weight_corr*corr_score(subset)
    return score

if __name__ =='__main__':
    #print('hello world')
    best_people, best_fitness, population = GA_SVM()
    print("特征子集：")
    print(best_people)
    acc, f1, precision, recall, roc_auc = pjzhunze(best_people)
    print("accuracy:%f" % acc)
    print("F1-Score:%f" % f1)
    print("Precision:%f" % precision)
    print("recall:%f" % recall)
    print("AUC值：%f" % roc_auc)

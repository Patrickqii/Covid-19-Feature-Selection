from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

data = pd.read_excel("/Users/patrickqi/Desktop/毕设/Dataset.xlsx")
data1 = data.iloc[:, 0:8]
data2 = np.mat(data1)
tezheng = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']

def jqSubset(tezheng, xulie):
    tz=[]
    for num in range(0, len(xulie)):
        if xulie[num]==1:
            tz.append(tezheng[num])
    return tz

def Jd(x, d1):
    # 从特征向量x中提取出相应的特征
    Feature = np.zeros(d1)  # 数组Feature用来存 x选择的是哪d1个特征
    k = 0
    for i in range(8):
        if x[i] == 1:
            Feature[k] = i
            k += 1

    # 将30个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3
    data3 = np.zeros((196, 1))
    for i in range(d1):
        p = Feature[i]
        p = p.astype(int)
        q = data2[:, p]
        q = q.reshape(196, 1)
        data3 = np.append(data3, q, axis=1)
    data3 = np.delete(data3, 0, axis=1)

    # 求类间离散度矩阵Sb
    data3_1 = data3[0:67, :]  # sonar数据集分为两类
    data3_2 = data3[67:196, :]
    m = np.mean(data3, axis=0)  # 总体均值向量
    m1 = np.mean(data3_1, axis=0)  # 第一类的均值向量
    m2 = np.mean(data3_2, axis=0)  # 第二类的均值向量
    m = m.reshape(d1, 1)  # 将均值向量转换为列向量以便于计算
    m1 = m1.reshape(d1, 1)
    m2 = m2.reshape(d1, 1)
    Sb = ((m1 - m).dot((m1 - m).T) * (67 / 196) + (m2 - m).dot((m2 - m).T) * (129 / 196))  # 除以类别个数

    # 求类内离散度矩阵Sw
    S1 = np.zeros((d1, d1))
    S2 = np.zeros((d1, d1))
    for i in range(67):
        S1 += (data3_1[i].reshape(d1, 1) - m1).dot((data3_1[i].reshape(d1, 1) - m1).T)
    S1 = S1 / 67
    for i in range(129):
        S2 += (data3_2[i].reshape(d1, 1) - m2).dot((data3_2[i].reshape(d1, 1) - m2).T)
    S2 = S2 / 129

    Sw = (S1 * (67 / 196) + S2 * (129 / 196))
    # Sw = (S1 + S2) / 2
    # 计算个体适应度函数 Jd(x)
    J1 = np.trace(Sb)
    J2 = np.trace(Sw)
    Jd = J1 / J2

    return Jd

def SFS(d):
    feature_selected = np.zeros(8)#每次根据Jd所选出来的特征的0，1矩阵
    fitness = np.zeros(8)
    #第一个特征的选择，遍历每一种特征的可能
    for i in range(8):
        tz_selection=np.zeros(8)
        tz_selection[i] = 1
        fitness[i] = Jd(tz_selection, 1)
    index = fitness.argmax()
    feature_selected[index] = 1

    #接下来每次向feature_selected中增加一个特征
    for i in range(2, d+1):
        fitness = np.zeros((9-i, 2))
        k = 0#k用来计数，最大是未被选择的特征数目
        for j in range(8):
            tz_selection = feature_selected.copy()
            if tz_selection[j] == 0:
                tz_selection[j] = 1
                fitness[k, 0] = j
                fitness[k, 1] = Jd(tz_selection, i)#此时选择的特征子集是tz_selected，所选特征数目是i
                k = k+1
        fitness = fitness[np.lexsort(fitness.T)]#将fitness按最后一列进行排序，也就是按照Jd进行排序，第一个就是我们所选择出的当前最优特征子集
        index = fitness[0, 0].astype(int)
        feature_selected[index] = 1

    return feature_selected

if __name__ == '__main__':
    best_subset = np.zeros(8)
    best_accuracy = 0
    for ii in range(1, 9):
        subset = SFS(ii)
        tz = jqSubset(tezheng, subset)
        x = data[tz]
        y = data['Target']
        #clf = svm.SVC(kernel='linear')
        #clf = GaussianNB()
        #clf = RandomForestClassifier(n_estimators=10)#基分类器数目为10
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
        accuracy = cross_val_score(clf, x, y, cv=5, scoring='accuracy').mean()
        print(subset)
        print(accuracy)
        if accuracy > best_accuracy:
            best_subset = subset.copy()
            best_accuracy = accuracy
    print("---------")
    tz = jqSubset(tezheng, best_subset)
    x = data[tz]
    y = data['Target']
    #clf = svm.SVC(kernel='linear')
    #clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=10)  # 基分类器数目为10
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
    score_f = cross_val_score(clf, x, y, cv=5, scoring='f1').mean()
    score_precision = cross_val_score(clf, x, y, cv=5, scoring='precision').mean()
    score_recall = cross_val_score(clf, x, y, cv=5, scoring='recall').mean()
    score_roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc').mean()
    print("特征子集：")
    print(best_subset)
    print("accuracy:%f" % best_accuracy)
    print("F1-Score:%f" % score_f)
    print("Precision:%f" % score_precision)
    print("recall:%f" % score_recall)
    print("AUC值：%f" % score_roc_auc)

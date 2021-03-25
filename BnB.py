from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def jqSubset(tezheng, xulie):
    tz=[]
    for num in range(0,len(xulie)):
        if xulie[num]==1:
            tz.append(tezheng[num])
    return tz

def BnB(xulie, i,tezheng,data):
    for num in range(i,len(xulie)):
        if xulie[num]==1:
            Nxulie=xulie[:]
            Nxulie[num]=0
            tz=jqSubset(tezheng,Nxulie)
            x=data[tz]
            y=data['Target']
            #clf = svm.SVC(kernel='linear')
            #clf = GaussianNB()
            #clf = RandomForestClassifier(n_estimators=10)#基分类器数目为10
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
            '''
            clf.fit(x, y)
            t = clf.predict(x)
            yy = y.values
            tt = t - yy
            score = (196 - np.count_nonzero(tt)) / 196
            '''
            score = cross_val_score(clf, x, y, cv=5, scoring='accuracy').mean()
            if score > 0.796538:
                score_f = cross_val_score(clf, x, y, cv=5, scoring='f1').mean()
                score_precision = cross_val_score(clf, x, y, cv=5, scoring='precision').mean()
                score_recall = cross_val_score(clf, x, y, cv=5, scoring='recall').mean()
                score_roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc').mean()
                csv_xulie = pd.DataFrame(
                    {'Feature 1': [Nxulie[0]], 'Feature 2': [Nxulie[1]], 'Feature 3': [Nxulie[2]], 'Feature 4': [Nxulie[3]],
                     'Feature 5': [Nxulie[4]], 'Feature 6': [Nxulie[5]], 'Feature 7': [Nxulie[6]], 'Feature 8': [Nxulie[7]],
                     'accuracy': [score], 'f1-score':[score_f], 'precision':[score_precision], 'recall':[score_recall], 'roc_auc':[score_roc_auc]})
                # df_empty=df_empty.append(csv_xulie)
                global  df_empty
                frames = [df_empty, csv_xulie]
                df_empty = pd.concat(frames)
                BnB(Nxulie,num+1,tezheng,data)

data = pd.read_excel("/Users/patrickqi/Desktop/毕设/Dataset.xlsx")
tezheng = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
xulie=[1,1,1,1,1,1,1,1]
df_empty=pd.DataFrame(columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'accuracy', 'f1-score', 'precision', 'recall', 'roc_auc'])
tz=jqSubset(tezheng,xulie)
x=data[tz]
y=data['Target']
# clf = svm.SVC(kernel='linear')
#clf = GaussianNB()
# clf = RandomForestClassifier(n_estimators=10)#基分类器数目为10
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
'''
clf.fit(x, y)
t = clf.predict(x)
yy = y.values
tt = t - yy
score = (196 - np.count_nonzero(tt)) / 196
'''
#cv_results = cross_validate(clf, x, y, scoring='accuracy')
#score = float(cv_results['test_score'].mean())
score = cross_val_score(clf, x, y, cv=5, scoring='accuracy').mean()
score_f = cross_val_score(clf, x, y, cv=5, scoring='f1').mean()
score_precision = cross_val_score(clf, x, y, cv=5, scoring='precision').mean()
score_recall = cross_val_score(clf, x, y, cv=5, scoring='recall').mean()
score_roc_auc = cross_val_score(clf, x, y, cv=5, scoring='roc_auc').mean()
csv_xulie = pd.DataFrame(
    {'Feature 1': [xulie[0]], 'Feature 2': [xulie[1]], 'Feature 3': [xulie[2]], 'Feature 4': [xulie[3]],
     'Feature 5': [xulie[4]], 'Feature 6': [xulie[5]], 'Feature 7': [xulie[6]], 'Feature 8': [xulie[7]],
     'accuracy': [score], 'f1-score':[score_f], 'precision':[score_precision], 'recall':[score_recall], 'roc_auc':[score_roc_auc]})
frames = [df_empty, csv_xulie]
df_empty = pd.concat(frames)
BnB(xulie,0,tezheng,data)

print(df_empty)
df_empty.to_csv("result1.csv", index=False, sep=',')
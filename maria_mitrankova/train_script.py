import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, roc_auc_score
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
df = pd.read_csv("dataset.csv")
X, y = df.iloc[:,:-1], df["target"]
print("counts of each class:", y.value_counts()) # датасет несбалансированный

# разобьем так, чтобы в train и test было пропорционально одинаково объектов классов
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) 
print("number of missing values:", df.isna().sum().sum()) # пропусков нет
# Сделаем upsampling тренировочной выборки
newdata = pd.concat([X_train, y_train], axis=1)
pos_X = newdata[newdata.iloc[:,-1] == 1]
neg_X = newdata[newdata.iloc[:,-1] == 0]

pos_upsample = resample(pos_X,
             replace=True,
             n_samples=neg_X.shape[0])
df_res = shuffle(np.concatenate([neg_X, pos_upsample]))
X_train, y_train = df_res[:,:-1], df_res[:, -1]
# в качестве baseline возьмем алгоритм Наивный Байес
gnb = GaussianNB()
y_pred_nb = gnb.fit(X_train, y_train).predict(X_test)
print("Naive Bayes roc auc on test:", roc_auc_score(y_test, y_pred_nb))

# Logistic Regression
# Data Normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)
# Сетка для подбора гиперпараметров
parameters = {'C':[0.1, 0.5, 1, 2], 'l1_ratio':[0, 0.5, 1]}
lreg = LogisticRegression(penalty="elasticnet", solver='saga', class_weight='balanced')
# Запускаем GridSearch:
print("LogReg: Starting GridSearch...")
clf_lr = GridSearchCV(lreg, parameters,scoring='roc_auc')
clf_lr.fit(X_tr_sc, y_train)
print("logReg best score on grid search:", clf_lr.best_score_)
print("logReg best hyperparams:", clf_lr.best_params_)
y_pred_lr = clf_lr.predict(X_test_sc)
print("roc auc for logReg on test:", roc_auc_score(y_test, y_pred_lr))
# # алгоритм SVM  -- считается довольно долго, поэтому закомментировала
# svm_parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 1, 5]}
# svc = SVC(class_weight='balanced')
# print("SVM: Starting GridSearch...")
# clf_svc = GridSearchCV(svc, svm_parameters,scoring='roc_auc', cv=3)
# clf_svc.fit(X_tr_sc, y_train)
# print("SVM best score on grid search:", clf_svc.best_score_)
# print("SVM best hyperparams:", clf_svc.best_params_)
# y_pred_svc = clf_svc.predict(X_test_sc)
# print("roc auc for SVM on test:", roc_auc_score(y_test, y_pred_svc))
# AdaBoost
random_grid = { 
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
}


abc = AdaBoostClassifier(n_estimators=200)
print("AdaBoost: Starting Grid Search...")
clf = GridSearchCV(abc, random_grid,scoring='roc_auc')
clf.fit(X_train, y_train)
print("adaboost best score on grid search:", clf.best_score_)
print("adaboost best params on grid search:", clf.best_params_)
y_pred = clf.predict(X_test)
print("adaboost roc auc on test:", roc_auc_score(y_test, y_pred))

# алгоритм Random Forest
# Сетка для подбора оптимальных гиперпараметров:
random_grid = { 
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,6,10],
    'criterion' :['gini', 'entropy'],
}

rf = RandomForestClassifier(n_estimators=300, random_state=42)
#Запустим GridSearch
print("Random Forest: Starting Grid Search...")
clf = GridSearchCV(rf, random_grid,scoring='roc_auc')
clf.fit(X_train, y_train)
print("random forest best score on grid search:", clf.best_score_)
print("random forest best hyperparams:", clf.best_params_)
proba = clf.predict_proba(X_test)

# Random forest показал наилучшие результаты, выбираем его и ищем оптимальный трешхолд
# оставляем вероятности для положительного класса
proba = proba[:, 1]
# вычислим pr auc curve
fpr, tpr, thresholds = roc_curve(y_test, proba)
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
y_pred = proba > thresholds[ix]
print("Random forest roc auc on test", roc_auc_score(y_test, y_pred))
# натренируем финальную модель на всех данных перед финальным предсказанием
clf.best_estimator_.fit(X,y)
# построим предсказание для тестовой выборки
final_test = pd.read_csv("test.csv")
proba = clf.predict_proba(final_test.iloc[:,1:])
y_pred_final = proba[:,1] > thresholds[ix]
# запишем в csv файл
df_test = pd.DataFrame(list(zip(final_test['id'], y_pred_final.astype(int))), columns=["id", "target"]).set_index('id')
df_test.to_csv("./submission.csv")

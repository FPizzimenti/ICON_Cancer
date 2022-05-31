import sklearn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pgmpy

from numpy import mean
from numpy import std 
from sklearn import model_selection 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

Cancer = pd.read_csv("C:\\Users\Rosalba\Desktop\icon\data.csv")
print()
print() 
print("Benvenuto nel nostro sistema per predire se, presi dei soggetti, essi sono affetti o meno dal Cancro al Seno")
print()
print()
print(Cancer) 
print() 
 
#dataset di input eliminando l'ultima colonna in quanto servirà per l'output 
X =Cancer.drop("diagnosis", axis=1) 
Y=Cancer["diagnosis"] 
 
# BILANCIAMENTO DELLE CLASSI 
# Proporzione dei non malati di tiroide (0) e malati di tiroide (1): [Numero di (non) malati di tiroide/Numero totale di pazienti] 
print() 
print('Pazienti non malati di cancro al seno:',Cancer.diagnosis.value_counts()[0], '(% {:.2f})'.format(Cancer.diagnosis.value_counts()[0] /Cancer.diagnosis.count() * 100)) 
print('Pazienti malati di cancro al seno:',Cancer.diagnosis.value_counts()[1], '(% {:.2f})'.format(Cancer.diagnosis.value_counts()[1] /Cancer.diagnosis.count() * 100), '\n') 
# Visualizzazione del grafico 
Cancer['diagnosis'].value_counts().plot(kind='bar').set_title('diagnosis') 
plt.show() 
 
# VERIFICA DELL'IMPORTANZA DELLE FEATURES
# Creazione della feature X e del target y
X = Cancer.drop('diagnosis', axis=1)
y = Cancer['diagnosis']

# Classificatore da utilizzare per la ricerca delle feature principali
rfc = RandomForestClassifier(random_state=42, n_estimators=100)
rfc_model = rfc.fit(X, y)

# Tracciamento delle feature in base alla loro importanza
(pd.Series(rfc_model.feature_importances_, index=X.columns)
    .nlargest(10)  # Numero massimo di feature da visualizzare
    .plot(kind='barh', figsize=[10,5])  # Tipo di grafico e dimensione
    .invert_yaxis())  # Assicuro un ordine decrescente

# Visualizzazione del grafico
plt.title('Top features derived by Random Forest', size=20)
plt.yticks(size=15)
plt.show()

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5)  # La classe è in squilibrio, quindi utilizzo Stratified K-Fold

#divissione dell'insieme di Training e l'insieme di Test, viene preso il 30% del Dataset per l'insieme di Test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) 
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3) 
print() 
print ("Xtrain") 
print (X_train) 
print() 
print ("Ytrain") 
print (Y_train) 
 
#knn 
knn= KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, Y_train) 
Y_pred = knn.predict(X_val) 
acc = accuracy_score(Y_pred, Y_val) 
prec = precision_score(Y_pred, Y_val) 
rec = recall_score(Y_pred, Y_val) 
f1 = f1_score(Y_pred, Y_val) 
classifier_name = "KNN" 
print() 
print("{}  {}:  {}".format(classifier_name, "accuracy", acc)) 
print("{}  {}:  {}".format(classifier_name, "precision", prec)) 
print("{}  {}:  {}".format(classifier_name, "recall", rec)) 
print("{}  {}:  {}".format(classifier_name, "F1", f1))
print()
scores_knn = cross_val_score(knn, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
acc=0.0 
prec=0.0 
rec=0.0 
f1=0.0
 
#Svm
from sklearn import svm
svm = svm.SVC(kernel='linear')
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_val)
acc=accuracy_score(Y_pred, Y_val)
prec = precision_score(Y_pred, Y_val)
rec = recall_score(Y_pred, Y_val)
f1 = f1_score(Y_pred, Y_val)
classifier_name = "SVM"
print("{}  {}:  {}".format(classifier_name, "accuracy", acc))
print("{}  {}:  {}".format(classifier_name, "precision", prec))
print("{}  {}:  {}".format(classifier_name, "recall", rec))
print("{}  {}:  {}".format(classifier_name, "F1", f1))
scores_svm = cross_val_score(svm, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
 
# Decision Tree 
dt = DecisionTreeClassifier(random_state = 10) 
dt.fit(X_train, Y_train) 
Y_pred = dt.predict(X_val) 
acc=accuracy_score(Y_pred, Y_val) 
prec = precision_score(Y_pred, Y_val) 
rec = recall_score(Y_pred, Y_val) 
f1 = f1_score(Y_pred, Y_val) 
classifier_name = "Decision Tree" 
print() 
print("{}  {}:  {}".format(classifier_name, "accuracy", acc)) 
print("{}  {}:  {}".format(classifier_name, "precision", prec)) 
print("{}  {}:  {}".format(classifier_name, "recall", rec)) 
print("{}  {}:  {}".format(classifier_name, "F1", f1)) 
scores_dt = cross_val_score(dt, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
acc=0.0 
prec=0.0 
rec=0.0 
f1=0.0  

#random forest 
rf = RandomForestClassifier(random_state = 23) 
rf.fit(X_train, Y_train) 
Y_pred = rf.predict(X_val) 
acc=accuracy_score(Y_pred, Y_val) 
prec = precision_score(Y_pred, Y_val) 
rec = recall_score(Y_pred, Y_val) 
f1 = f1_score(Y_pred, Y_val) 
classifier_name = "Random Forest" 
print() 
print("{}  {}:  {}".format(classifier_name, "accuracy", acc)) 
print("{}  {}:  {}".format(classifier_name, "precision", prec)) 
print("{}  {}:  {}".format(classifier_name, "recall", rec)) 
print("{}  {}:  {}".format(classifier_name, "F1", f1))
scores_rf = cross_val_score(rf, X, Y, scoring='accuracy', cv=kf, n_jobs=-1)
print()

#confronto tra le metriche
print('KNN K-fold: %.3f (%.3f)' % (mean(scores_knn), std(scores_knn)))
print('SVM K-fold: %.3f (%.3f)' % (mean(scores_svm), std(scores_svm)))
print('Decision Tree K-fold: %.3f (%.3f)' % (mean(scores_dt), std(scores_dt)))
print('Random Forest K-fold: %.3f (%.3f)' % (mean(scores_rf), std(scores_rf)))
print()

# CREAZIONE DELLA RETE BAYESIANA
# Conversione di tutti i valori all'interno del dataframe in interi
Cancer_int = np.array(Cancer, dtype=int)
Cancer = pd.DataFrame(Cancer_int, columns=Cancer.columns)

# Creazione della feature X e del target y
X_train = Cancer
y_train = Cancer["diagnosis"]

# Creazione della struttura della rete
k2 = K2Score(X_train)
hc_k2 = HillClimbSearch(X_train)
k2_model = hc_k2.estimate(scoring_method=k2)

# Creazione della rete
bNet = BayesianNetwork(k2_model.edges())
bNet.fit(Cancer, estimator=MaximumLikelihoodEstimator)

# Visualizzazione dei nodi e degli archi
print('\033[1m' + '\nNodi della rete:\n' + '\033[0m', bNet.nodes)
print('\033[1m' + '\nArchi della rete:\n' + '\033[0m', bNet.edges)

# CALCOLO DELLA PROBABILITÀ

# Calcolo della probabilità per un soggetto presumibilmente senza cancro (0) ed uno con cancro (1) di avere il cancro al seno

# Eliminazione delle variabili ininfluenti
data = VariableElimination(bNet)

# Soggetto potenzialmente senza cancro al seno
notCancer= data.query(variables = ['diagnosis'],
                         evidence = { 'radius_mean':14,'texture_mean':16,'perimeter_mean':91,'area_mean':334,
                                      'radius_se':0,'perimeter_se':1,'area_se':2,'radius_worst':16,'texture_worst':25,'perimeter_worst':106,
                                      'area_worst':520,'compactness_worst':0,'concavity_worst':0 })

print('\nProbabilità per un soggetto potenzialmente senza cancro al seno:')
print(notCancer, '\n')

# Test su Soggetto potenzialmente senza cancro al seno
TestnotCancer= data.query(variables = ['diagnosis'],
                         evidence = { 'radius_mean':14,'texture_mean':16,'perimeter_mean':91,'area_mean':334,
                                      'radius_se':0,'perimeter_se':5,'area_se':20,'radius_worst':16,'texture_worst':25,'perimeter_worst':106,
                                      'area_worst':520,'compactness_worst':0,'concavity_worst':0 })

print('\nTest su un soggetto potenzialmente senza cancro al seno:')
print(TestnotCancer, '\n')

# Soggetto potenzialmente con cancro al seno
cancer = data.query(variables = ['diagnosis'],
                       evidence={'radius_mean': 15,'texture_mean':23,'perimeter_mean':80,'area_mean':301,
                       'radius_se':0,'perimeter_se':4,'area_se':30, 'radius_worst':15,'texture_worst':32,'perimeter_worst':170,
                       'area_worst':406, 'compactness_worst':0,'concavity_worst':0})

print('\nProbabilità per un soggetto potenzialmente con cancro al seno:')
print(cancer)

# Test su Soggetto potenzialmente con cancro al seno
Testcancer = data.query(variables = ['diagnosis'],
                       evidence={'radius_mean': 15,'texture_mean':23,'perimeter_mean':80,'area_mean':301,
                       'radius_se':0,'perimeter_se':1,'area_se':10, 'radius_worst':15,'texture_worst':32,'perimeter_worst':170,
                       'area_worst':406, 'compactness_worst':0,'concavity_worst':0})

print('\n Test su un soggetto potenzialmente con cancro al seno:')
print(Testcancer)



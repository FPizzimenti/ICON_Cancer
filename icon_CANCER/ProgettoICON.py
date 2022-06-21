# Progetto Ingegneria della Conoscenza
# Autori: Nardiello Rosalba e Pizzimenti Francesca 

import sklearn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pgmpy

from numpy import mean
from numpy import std 
from sklearn import model_selection 
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
from sklearn import svm
from sklearn.svm import SVC

Cancer = pd.read_csv("C:\\Users\Rosalba\Desktop\icon_CANCER\data.csv")
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

X = Cancer.to_numpy()
y = Cancer["diagnosis"].to_numpy()

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5)  # La classe è in squilibrio, quindi utilizzo Stratified K-Fold

# Classificatori per la valutazione
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()

# Score delle metriche
model = {
        'KNN' : {'accuracy_list': 0.0,
                 'precision_list' : 0.0,
                 'recall_list' : 0.0,
                 'f1_list' : 0.0
        },

        'DecisionTree' : {'accuracy_list': 0.0,
                                    'precision_list' : 0.0,
                                    'recall_list' : 0.0,
                                    'f1_list' : 0.0
        },

        'RandomForest' : {'accuracy_list': 0.0,
                                    'precision_list' : 0.0,
                                    'recall_list' : 0.0,
                                    'f1_list' : 0.0
        },

        'SVM' : {'accuracy_list': 0.0,
                 'precision_list' : 0.0,
                 'recall_list' : 0.0,
                 'f1_list' : 0.0
        }
}

# K-Fold dei classificatori
for train_index, test_index in kf.split(X, y):

    training_set, testing_set = X[train_index], X[test_index]

    # Dati di train
    data_train = pd.DataFrame(training_set, columns=Cancer.columns)
    X_train = data_train.drop("diagnosis", axis=1)
    y_train = data_train.diagnosis

    # Dati di test
    data_test = pd.DataFrame(testing_set, columns=Cancer.columns)
    X_test = data_test.drop("diagnosis", axis=1)
    y_test = data_test.diagnosis

    # Fit dei classificatori
    knn.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_SVM = svc.predict(X_test)

    # Salvo le metriche del fold nel dizionario
    model['KNN']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_knn))
    model['KNN']['precision_list'] = (metrics.precision_score(y_test, y_pred_knn))
    model['KNN']['recall_list'] = (metrics.recall_score(y_test,y_pred_knn))
    model['KNN']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

    model['DecisionTree']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_dtc))
    model['DecisionTree']['precision_list'] = (metrics.precision_score(y_test, y_pred_dtc))
    model['DecisionTree']['recall_list'] = (metrics.recall_score(y_test, y_pred_dtc))
    model['DecisionTree']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

    model['RandomForest']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_rfc))
    model['RandomForest']['precision_list'] = (metrics.precision_score(y_test, y_pred_rfc))
    model['RandomForest']['recall_list'] = (metrics.recall_score(y_test, y_pred_rfc))
    model['RandomForest']['f1_list'] = (metrics.f1_score(y_test, y_pred_rfc))

    model['SVM']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_SVM))
    model['SVM']['precision_list'] = (metrics.precision_score(y_test, y_pred_SVM))
    model['SVM']['recall_list'] = (metrics.recall_score(y_test, y_pred_SVM))
    model['SVM']['f1_list'] = (metrics.f1_score(y_test, y_pred_SVM))

# Modello di rapporto
def model_report(model):

    Cancer_models = []

    for clf in model:
        Cancer_model = pd.DataFrame({'model'        : [clf],
                                 'accuracy'     : [np.mean(model[clf]['accuracy_list'])],
                                 'precision'    : [np.mean(model[clf]['precision_list'])],
                                 'recall'       : [np.mean(model[clf]['recall_list'])],
                                 'f1score'      : [np.mean(model[clf]['f1_list'])]
                                 })
        
        Cancer_models.append(Cancer_model)
        
    return Cancer_models

# Visualizzazione della tabella con le metriche
Cancer_models_concat = pd.concat(model_report(model), axis=0).reset_index()  # Concatenazione dei modelli
Cancer_models_concat = Cancer_models_concat.drop('index', axis=1)  # Rimozione dell'indice
print(Cancer_models_concat)  # Visualizzazione della tabella

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



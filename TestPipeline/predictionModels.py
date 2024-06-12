from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
# from pulearn import ElkanotoPuClassifier
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def get_classifiers():
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=42),
        # XGBClassifier(n_jobs=-1, random_state=42),
        # LinearSVC(random_state=42,max_iter=5000),
        # LogisticRegression(n_jobs=-1, random_state=42),
        # GaussianNB(),
        # CatBoostClassifier(random_state=42,logging_level='Silent'),
        # SGDClassifier(n_jobs=-1, random_state=42)
    ]
    return classifiers

embedCSV = pd.read_csv('../DB/HuriTest/embeddings_train.csv',index_col=0)
embeddings_array = np.array(list(embedCSV.values))
dict_embeddings = {embedCSV.index[i]: embeddings_array[i] for i in range(len(embeddings_array))}
vector_size = embedCSV.shape[1]

for t in ['Static']: #'Growing', 'Undersampling', , 'Uniform'
    logging.info("Starting training with " + t + " dataset")
    for i in range(800, -1, -200):
        logging.info("Starting training with threshold: " + str(i))
        metrics_to_csv = {}
        for f in range(10):
            logging.info("Starting training with fold: " + str(f))
            X_train1 = np.loadtxt('../DB/'+t+'/T'+str(i)+'/X_train_Fold'+str(f)+'.csv', delimiter=',',dtype=str)
            y_train = np.loadtxt('../DB/'+t+'/T'+str(i)+'/y_train_Fold'+str(f)+'.csv', delimiter=',')
            X_test1 = np.loadtxt('../DB/'+t+'/T'+str(i)+'/X_test_Fold'+str(f)+'.csv', delimiter=',',dtype=str)
            y_test = np.loadtxt('../DB/'+t+'/T'+str(i)+'/y_test_Fold'+str(f)+'.csv', delimiter=',')
            sample_weight = np.loadtxt('../DB/'+t+'/T'+str(i)+'/sample_weight_train_Fold'+str(f)+'.csv', delimiter=',')
            sample_weight = np.array([float(val/1000) for val in sample_weight])
            X_train = []
            for prot1, prot2 in X_train1:
                prot1 = 'https://string-db.org/network/9606.ENSP' + prot1
                prot2 = 'https://string-db.org/network/9606.ENSP' + prot2
                emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
                emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
                hada = np.multiply(emb_prot1, emb_prot2)
                X_train.append(hada.tolist()[0])
            
            X_test = []
            for prot1, prot2 in X_test1:
                prot1 = 'https://string-db.org/network/9606.ENSP' + prot1
                prot2 = 'https://string-db.org/network/9606.ENSP' + prot2
                emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
                emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
                hada = np.multiply(emb_prot1, emb_prot2)
                X_test.append(hada.tolist()[0])

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            for m in ['Normal','Weighted']: # ,'PU', 
                if m not in metrics_to_csv:
                    metrics_to_csv[m] = {}
                logging.info("Starting training with model: " + m)
                clfs = get_classifiers()
                for c in clfs:
                    n = type(c).__name__
                    if n not in metrics_to_csv[m]:
                        metrics_to_csv[m][n] = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])
                    if m == 'Weighted':
                        c.fit(X_train, y_train, sample_weight = sample_weight)
                        y_pred = c.predict(X_test)
                    # elif m == 'PU':
                    #     pu = ElkanotoPuClassifier(estimator = c, hold_out_ratio = 0.1)
                    #     y_train_pu = np.where(y_train == 0, -1, y_train)
                    #     pu.fit(X_train, y_train)
                    #     y_pred = pu.predict(X_test)
                    else:
                        c.fit(X_train, y_train)
                        y_pred = c.predict(X_test)
                    prec = precision_score(y_test, y_pred, average='weighted')
                    rec = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred)
                    metrics_to_csv[m][n].loc[len(metrics_to_csv[m][n].index)] = [prec, rec, f1, acc, auc]
                    if f == 9:
                        Path('../Results/12_06Test/Folds/').mkdir(parents=True, exist_ok=True)
                        metrics_to_csv[m][n].to_csv('../Results/12_06Test/Folds/'+t+'T'+str(i)+'metrics_'+m+n+'.csv', index=False)
                
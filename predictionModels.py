from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

classifiers = [
    RandomForestClassifier(n_jobs=-1, random_state=42),
    XGBClassifier(n_jobs=-1, random_state=42),
]

for t in ['Uniform','Static','Growing']:
    for i in range(800, -1, -200):
        for m in ['Normal','Weighted','PU']:
            for c in classifiers:
                metrics_to_csv = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])
                for f in range(10):
                    X_train = np.loadtxt('DB/'+t+'/T'+str(i)+'/X_train_Fold'+str(f)+'.csv', delimiter=',')
                    y_train = np.loadtxt('DB/'+t+'/T'+str(i)+'/y_train_Fold'+str(f)+'.csv', delimiter=',')
                    X_test = np.loadtxt('DB/'+t+'/T'+str(i)+'/X_test_Fold'+str(f)+'.csv', delimiter=',')
                    y_test = np.loadtxt('DB/'+t+'/T'+str(i)+'/y_test_Fold'+str(f)+'.csv', delimiter=',')
                    sample_weight = np.loadtxt('DB/'+t+'/T'+str(i)+'/sample_weight_Fold'+str(f)+'.csv', delimiter=',')
                    if m == 'Weighted':
                        c.fit(X_train, y_train, sample_weight = sample_weight)
                        y_pred = c.predict(X_test)
                    elif m == 'PU':
                        pu = ElkanotoPuClassifier(estimator = c, hold_out_ratio = 0.1)
                        y_train_pu = np.where(y_train == 0, -1, y_train)
                        pu.fit(X_train, y_train)
                        y_pred = pu.predict(X_test)
                    else:
                        c.fit(X_train, y_train)
                        y_pred = c.predict(X_test)
                    prec = precision_score(y_test, y_pred, average='weighted')
                    rec = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred)
                    metrics_to_csv = metrics_to_csv.append({'precision': prec, 'recall': rec, 'WAF': f1, 'accuracy': acc, 'AUC': auc}, ignore_index=True)
                if m == 'PU':
                    metrics_to_csv.to_csv('Results/7_03Test/'+t+'T'+str(i)+'metrics_'+m+c+'.csv', index=False)
                else:
                    metrics_to_csv.to_csv('Results/7_03Test/'+t+'T'+str(i)+'metrics_'+m+c+'.csv', index=False)
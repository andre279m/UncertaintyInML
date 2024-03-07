from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

classifiers = [
    RandomForestClassifier(n_jobs=-1, random_state=42),
    XGBClassifier(n_jobs=-1, random_state=42),
]

for t in ['Uniform','Static','Growing']:
    for i in range(800, -1, -200):
        for f in range(10):
            X_train = np.loadtxt('DB/X_train_'+t+'_'+str(i)+'_'+str(f)+'.csv', delimiter=',')
            X_test
            y_train
            y_test
            sample_weight
            for m in ['Normal','Weighted','PU']:
                for c in classifiers:
                
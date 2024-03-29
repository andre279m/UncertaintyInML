# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import random,logging
from pathlib import Path
from pulearn import ElkanotoPuClassifier
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def get_classifiers():
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=42),
        XGBClassifier(n_jobs=-1, random_state=42),
    ]
    return classifiers

# Files
protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'

# Creating the Knowledge graph
prots = pd.read_csv(protein_file_path, sep='\t', header=0)
prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
prots = prots['#string_protein_id'].unique().tolist()

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)]
# mean of confidence score
mean = data_full['combined_score'].mean()

data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True)

setsize = int(len(data8)*5)

data = data_full.sample(n=setsize, random_state=42).dropna().reset_index(drop=True)

# Without semantic similarity
all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

del data_full
del prots
# Pipeline
embedCSV = pd.read_csv('DB/embeddings.csv',index_col=0)
embeddings_array = np.array(list(embedCSV.values))
dict_embeddings = {embedCSV.index[i]: embeddings_array[i] for i in range(len(embeddings_array))}
vector_size = embedCSV.shape[1]
random.seed(42)

f1_to_csv = {}

pairs_prots = []
for d in data.values:
    pairs_prots.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1], d[-1]))

negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))

X, y, sample_weight = [], [], []
for prot1, prot2, c in pairs_prots:
    prot1 = 'https://string-db.org/network/9606.ENSP' + prot1
    prot2 = 'https://string-db.org/network/9606.ENSP' + prot2
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X.append(hada.tolist()[0])
    y.append(1)
    sample_weight.append(c)

for prot1, prot2, label in negative_pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X.append(hada.tolist()[0])
    y.append(0)
    sample_weight.append(mean)

metrics_to_csv = {}

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
X, y ,sample_weight = np.array(X), np.array(y), np.array(sample_weight)
for j, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sample_weight_train = sample_weight[train_index]

    logging.info("Training with fold: " + str(j))

    for m in ['PU','Normal','Weighted']:
        if m not in metrics_to_csv:
            metrics_to_csv[m] = {}
        logging.info("Starting training with model: " + m)
        clfs = get_classifiers()
        for c in clfs:
            n = type(c).__name__
            if n not in metrics_to_csv[m]:
                metrics_to_csv[m][n] = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])
            if m == 'Weighted':
                c.fit(X_train, y_train, sample_weight = sample_weight_train)
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
            metrics_to_csv[m][n].loc[len(metrics_to_csv[m][n].index)] = [prec, rec, f1, acc, auc]
            if j == 9:
                Path('Results/22_03Test/Random/').mkdir(parents=True, exist_ok=True)
                metrics_to_csv[m][n].to_csv('Results/22_03Test/Random/'+'metrics_'+m+n+'.csv', index=False)
        
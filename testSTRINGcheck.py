# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import random,logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def get_classifiers():
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=42),
        # XGBClassifier(n_jobs=-1, random_state=42),
    ]
    return classifiers

# Files
protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'
semantic_similarity_file_path = 'DB/NegativeSamplesUncertainty.csv'
# # Creating the Knowledge graph
prots = pd.read_csv(protein_file_path, sep='\t', header=0)
prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
prots = prots['#string_protein_id'].unique().tolist()

# # Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)]
# mean of confidence score
# mean = data_full['combined_score'].mean()

all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

negative_pairs_prots_test = pd.read_csv('DB/HuriTest/HuRI_negatives.tsv',sep='\t',header=None)
negative_pairs_prots_test = list(('http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + x[0], 'http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + x[1],0) for x in negative_pairs_prots_test.to_numpy())

# Pipeline
embedtrainCSV = pd.read_csv('DB/HuriTest/embeddings_train.csv',index_col=0)
embeddings_train_array = np.array(list(embedtrainCSV.values))
train_embeddings = {embedtrainCSV.index[i]: embeddings_train_array[i] for i in range(len(embeddings_train_array))}

embedtestCSV = pd.read_csv('DB/HuriTest/embeddings_test.csv',index_col=0)
embeddings_test_array = np.array(list(embedtestCSV.values))
test_embeddings = {embedtestCSV.index[i]: embeddings_test_array[i] for i in range(len(embeddings_test_array))}

vector_size = embedtrainCSV.shape[1]
random.seed(42)

huri_data = pd.read_csv('DB/HuriTest/HuRI.tsv',sep='\t',header=None,names=['protein1', 'protein2'])

huri_prots = np.loadtxt('DB/HuriTest/HuRI_proteins.txt',dtype=str)
huri_data = huri_data[huri_data["protein1"].isin(huri_prots) & huri_data["protein2"].isin(huri_prots)].reset_index(drop=True)

pairs_prots_huri = []
for d in huri_data.values:
    pairs_prots_huri.append(('http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + d[0],'http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + d[1], 1))

dataSample = data_full.sample(n=len(pairs_prots_huri),random_state=42)

pairs_prots = []
for d in dataSample.values:
    pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

pairs_prots = pairs_prots + pairs_prots_huri
dict_embeddings = train_embeddings
dict_embeddings.update(test_embeddings)

negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots)-len(negative_pairs_prots_test))
negative_pairs_prots = negative_pairs_prots + negative_pairs_prots_test
f1_to_csv = {}

X, y = [], []
for prot1, prot2, label in pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X.append(hada.tolist()[0])
    y.append(int(label))

for prot1, prot2, label in negative_pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X.append(hada.tolist()[0])
    y.append(0)

metrics_to_csv = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])

# Creating training set and test set
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
X, y = np.array(X), np.array(y)
for j, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # logging.info("Training with threshold: " + str(i) + " and fold: " + str(j))
    clfs = get_classifiers()
    for c in clfs:
        n = type(c).__name__
        m = 'Normal'
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        metrics_to_csv.loc[j] = [prec, rec, f1, acc, auc]
        if j == 9:
            Path('Results/5_04Test/HuRICheck/').mkdir(parents=True, exist_ok=True)
            metrics_to_csv.to_csv('Results/5_04Test/HuRICheck/Mixed50Metrics_'+m+n+'.csv', index=False)
        
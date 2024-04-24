# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
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

# Creating thresholds
# Thresholds for the whole dataset

# data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True)
# data6 = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True)
# data4 = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True)
# data2 = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True)
# dataSample = data_full

# Thresholds for fixed size sampling (1000 samples)

# data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(n=1000, random_state=42)
# data6 = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True).sample(n=1000,random_state=42)
# data4 = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True).sample(n=1000,random_state=42)
# data2 = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True).sample(n=1000,random_state=42)
# dataSample = data_full.sample(n=1000,random_state=42)

# Thresholds for fraction sampling (10% of the data)

# data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(frac=0.1, random_state=42)
# data6 = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
# data4 = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
# data2 = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
dataSample = data_full.sample(frac=0.1,random_state=42)


# Thresholds for stratified sampling
# setsize = int(len(data8)*0.1)
# select = [int((N*len(data8))/setsize) for N in range(setsize)]
# data8 = data8.loc[select]
# setsize = int(len(data6)*0.1)
# select = [int((N*len(data6))/setsize) for N in range(setsize)]
# data6 = data6.loc[select]
# setsize = int(len(data4)*0.1)
# select = [int((N*len(data4))/setsize) for N in range(setsize)]
# data4 = data4.loc[select]
# setsize = int(len(data2)*0.1)
# select = [int((N*len(data2))/setsize) for N in range(setsize)]
# data2 = data2.loc[select]
# setsize = int(len(dataSample)*0.1)
# select = [int((N*len(dataSample))/setsize) for N in range(setsize)]
# dataSample = dataSample.reset_index(drop=True).loc[select]


# Thresholds for training with data above 800 and testing with data below 800
# train = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
# test = data_full.where(data_full['combined_score']<800).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)

# Without semantic similarity
all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv',sep=',',header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

# With semantic similarity
# for n in negatives.values:
#     all_negative_pairs_prots.append((n[1],n[2],0))
#
# del negatives

# del data_full
# del prots
# Pipeline
embedCSV = pd.read_csv('DB/HuriTest/embeddings_train.csv',index_col=0)
embeddings_array = np.array(list(embedCSV.values))
dict_embeddings = {embedCSV.index[i]: embeddings_array[i] for i in range(len(embeddings_array))}
vector_size = embedCSV.shape[1]
random.seed(42)

f1_to_csv = {}

# for i in range(800, -1, -200):
#     if i == 800:
#         data = data8
#     elif i == 600:
#         data = data6
#     elif i == 400:
#         data = data4
#     elif i == 200:
#         data = data2
#     elif i == 0:
#         data = dataSample

# huri_data = pd.read_csv('DB/HuriTest/HuRI.tsv',sep='\t',header=None,names=['protein1', 'protein2'])


# huri_prots = np.loadtxt('DB/HuriTest/HuRI_proteins.txt',dtype=str)
# huri_data = huri_data[huri_data["protein1"].isin(huri_prots) & huri_data["protein2"].isin(huri_prots)].reset_index(drop=True)

pairs_prots = []
for d in dataSample.values:
    pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

# Without semantic similarity
negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))
# With semantic similarity
#negative_pairs_prots = all_negative_pairs_prots[:len(pairs_prots)]    

# # Sample weight
# sample_weight = list(data['combined_score'].values)
# sample_weight.extend([mean for i in range(0, len(pairs_prots))])
# sample_weight = np.array(sample_weight)

# sample_weight = np.exp(0.1 * (sample_weight - 700))

# Generating pair representations using hadamard operator # other possibilities are concatenation, wl-1 or wl-2
X_train, y_train = [], []
for prot1, prot2, label in pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_train.append(hada.tolist()[0])
    y_train.append(int(label))

for prot1, prot2, label in negative_pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_train.append(hada.tolist()[0])
    y_train.append(0)

X_test1 = np.loadtxt('DB/Uniform/T0/X_test_Fold9.csv', delimiter=',',dtype=str)

X_test, y_test = [], []
for prot1, prot2 in X_test1:
    prot1 = 'https://string-db.org/network/9606.ENSP' + str(prot1)
    prot2 = 'https://string-db.org/network/9606.ENSP' + str(prot2)
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_test.append(hada.tolist()[0])
y_test = np.loadtxt('DB/Uniform/T0/y_test_Fold9.csv', delimiter=',')


metrics_to_csv = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])
# Creating training set and test set


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
    metrics_to_csv.loc[800] = [prec, rec, f1, acc, auc]
    Path('Results/5_04Test/HuRICheck/').mkdir(parents=True, exist_ok=True)
    metrics_to_csv.to_csv('Results/5_04Test/HuRICheck/metrics0_'+m+n+'.csv', index=False)

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

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
# Files
gene_ontology_file_path = 'DB/go.owl'
protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
protein_links_file_path = 'DB/9606.protein.links.v12.0.txt'
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'
semantic_similarity_file_path = 'DB/NegativeSamplesUncertainty.csv'
gene_ontology_annotated_file_path = 'DB/go_annotated.owl'
# Creating the Knowledge graph
prots = pd.read_csv(protein_file_path, sep='\t', header=0)
prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
prots = prots['#string_protein_id'].unique().tolist()

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)].sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)
# mean of confidence score
mean = data_full['combined_score'].mean()

# Creating thresholds
fulldata8 = data_full[data_full['combined_score'] > 800]
setsize = int(len(fulldata8)/2)
select = [int((N*len(data_full))/setsize) for N in range(setsize)]
train_data = data_full.loc[select]
dt8 = data_full[~data_full.isin(train_data)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)

testsize = int(len(data_full)*0.1)
select = [int((N*len(dt8))/testsize) for N in range(testsize)]
test_data8 = dt8.loc[select]

# Without semantic similarity
all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

del data_full
del prots
del fulldata8
del dt8
# Pipeline
embedCSV = pd.read_csv('DB/embeddings.csv',index_col=0)
embeddings_array = np.array(list(embedCSV.values))
dict_embeddings = {embedCSV.index[i]: embeddings_array[i] for i in range(len(embeddings_array))}
vector_size = embedCSV.shape[1]
random.seed(42)

# List of classifiers
classifiers = [
    RandomForestClassifier(n_jobs=-1, random_state=42),
#     BaggingClassifier(random_state=42, n_jobs=-1),
    XGBClassifier(n_jobs=-1, random_state=42),
#     GaussianNB()
#    AdaBoostClassifier(random_state=42)
]

metrics_to_csv = pd.DataFrame(columns=['precision', 'recall', 'WAF'])

data = train_data

pairs_prots = []
for d in data.values:
    pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

# Without semantic similarity
negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))
# With semantic similarity
#negative_pairs_prots = all_negative_pairs_prots[:len(pairs_prots)]    

# Sample weight
sample_weight = list(data['combined_score'].values)
sample_weight.extend([100 for i in range(0, len(pairs_prots))])
sample_weight = np.array(sample_weight)

# sample_weight = np.log(sample_weight)

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
    y_train.append(int(label))

pairs_prots = []
for d in test_data8.values:
    pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

X_test, y_test = [], []
for prot1, prot2, label in pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_test.append(hada.tolist()[0])
    y_test.append(int(label))

negative_pairs_prots = set(negative_pairs_prots)

npp = [x for x in all_negative_pairs_prots if x not in negative_pairs_prots]

negative_pairs_prots = random.sample(npp, len(pairs_prots))

for prot1, prot2, label in negative_pairs_prots:
    emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_test.append(hada.tolist()[0])
    y_test.append(int(label))

for clf in classifiers:

    logging.info("Training with classifier: " + type(clf).__name__)

    clf.fit(X_train, y_train)
    # Obtaining predictions
    pred_test = clf.predict(X_test)
    # Computing performance metrics
    weighted_avg_f1 = metrics.f1_score(y_test, pred_test, average='weighted')
    weighted_avg_precision = metrics.precision_score(y_test, pred_test, average='weighted')
    weighted_avg_recall = metrics.recall_score(y_test, pred_test, average='weighted')     

    n = type(clf).__name__
    metrics_to_csv.loc[n] = [weighted_avg_precision, weighted_avg_recall, weighted_avg_f1]

metrics_to_csv.to_csv('Results/metrics_results_trainWabove800WSampleWeight100.csv')
        

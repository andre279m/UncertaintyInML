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
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'
semantic_similarity_file_path = 'DB/NegativeSamplesUncertainty.csv'
gene_ontology_annotated_file_path = 'DB/go_annotated2.owl'
# Creating the Knowledge graph
prots = pd.read_csv(protein_file_path, sep='\t', header=0)
prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
prots = prots['#string_protein_id'].unique().tolist()

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)]
# mean of confidence score
mean = data_full['combined_score'].mean()

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

data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(frac=0.1, random_state=42)
data6 = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
data4 = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
data2 = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
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
all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

# With semantic similarity
# for n in negatives.values:
#     all_negative_pairs_prots.append((n[1],n[2],0))
#
# del negatives

del data_full
del prots
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

f1_to_csv = {}

for i in range(800, -1, -200):
    if i == 800:
        data = data8
    elif i == 600:
        data = data6
    elif i == 400:
        data = data4
    elif i == 200:
        data = data2
    elif i == 0:
        data = dataSample

    pairs_prots = []
    for d in data.values:
        pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

    # Without semantic similarity
    negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))
    # With semantic similarity
    #negative_pairs_prots = all_negative_pairs_prots[:len(pairs_prots)]    

    # Sample weight
    sample_weight = list(data['combined_score'].values)
    sample_weight.extend([mean for i in range(0, len(pairs_prots))])
    sample_weight = np.array(sample_weight)

    # sample_weight = np.exp(0.1 * (sample_weight - 700))

    # Generating pair representations using hadamard operator # other possibilities are concatenation, wl-1 or wl-2
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
        y.append(int(label))

    # Creating training set and test set
    skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    X, y = np.array(X), np.array(y)
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sample_weight_train = sample_weight[train_index]

        logging.info("Training with threshold: " + str(i) + " and fold: " + str(j))

        for clf in classifiers:

            logging.info("Training with classifier: " + type(clf).__name__)

            clf.fit(X_train, y_train)
            # Obtaining predictions
            pred_test = clf.predict(X_test)
            # Computing performance metrics
            weighted_avg_f1 = metrics.f1_score(y_test, pred_test, average='weighted')
            
            n = type(clf).__name__
            if n not in f1_to_csv :
                f1_to_csv[n] = {}   

            if i not in f1_to_csv[n] :
                f1_to_csv[n][i] = [weighted_avg_f1]
            else :
                f1_to_csv[n][i].append(weighted_avg_f1)
    
for v in f1_to_csv.keys():
    df = pd.DataFrame.from_dict(f1_to_csv[v], orient='index')
    df.to_csv('Results/f1_results_negsWexp01_' + v + '_Fraction10.csv')
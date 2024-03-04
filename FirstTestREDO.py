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

gene_ontology_file_path = 'DB/go.owl'
protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
protein_links_file_path = 'DB/9606.protein.links.v12.0.txt'
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'
semantic_similarity_file_path = 'DB/NegativeSamplesUncertainty.csv'
gene_ontology_annotated_file_path = 'DB/go_annotated.owl'

embedCSV = pd.read_csv('DB/embeddings.csv',index_col=0)
embeddings_array = np.array(list(embedCSV.values))
dict_embeddings = {embedCSV.index[i]: embeddings_array[i] for i in range(len(embeddings_array))}
vector_size = embedCSV.shape[1]
random.seed(42)

classifiers = [
    RandomForestClassifier(n_jobs=-1, random_state=42),
#     BaggingClassifier(random_state=42, n_jobs=-1),
    XGBClassifier(n_jobs=-1, random_state=42),
#     GaussianNB()
#    AdaBoostClassifier(random_state=42)
]

f1_to_csv = {}

for i in range(8, -1, -2):
    train_data = pd.read_csv("DB/TRAIN/train_data" + str(i) + ".csv", sep=",", header=0)

    pairs_prots = []
    for d in train_data.values:
        pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))
    
    negative_train = pd.read_csv("DB/TRAIN/negative_train" + str(i) + ".csv", sep=",", header=0).values

    for j in range(5):

        test_data8 = pd.read_csv("DB/TEST/test_data" + str(j) + ".csv", sep=",", header=0)
        negative_test = pd.read_csv("DB/TEST/negative_pairs_prots" + str(j) + ".csv", sep=",", header=0).values

        # Sample weight
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

        for prot1, prot2, label in negative_train:
            emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
            emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
            hada = np.multiply(emb_prot1, emb_prot2)
            X_train.append(hada.tolist()[0])
            y_train.append(int(label))

        pairs_prots_test = []
        for d in test_data8.values:
            pairs_prots_test.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))

        X_test, y_test = [], []
        for prot1, prot2, label in pairs_prots_test:
            emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
            emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
            hada = np.multiply(emb_prot1, emb_prot2)
            X_test.append(hada.tolist()[0])
            y_test.append(int(label))

        for prot1, prot2, label in negative_test:
            emb_prot1 = dict_embeddings[prot1].reshape(1, vector_size)
            emb_prot2 = dict_embeddings[prot2].reshape(1, vector_size)
            hada = np.multiply(emb_prot1, emb_prot2)
            X_test.append(hada.tolist()[0])
            y_test.append(int(label))

        #sample_weight_train = sample_weight[train_index]

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
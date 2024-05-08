# Imports
import pandas as pd
import numpy as np
import random,logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pulearn import ElkanotoPuClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def get_classifiers():
    classifiers = [
        RandomForestClassifier(n_jobs=-1, random_state=42),
        XGBClassifier(n_jobs=-1, random_state=42),
    ]
    return classifiers
# Files
protein_file_path = '../DB/9606.protein.enrichment.terms.v12.0.txt'
protein_full_links_file_path = '../DB/9606.protein.links.detailed.v12.0.txt'
# Creating the Knowledge graph
prots = pd.read_csv(protein_file_path, sep='\t', header=0)
prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
prots = prots['#string_protein_id'].unique().tolist()

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)].sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)

mean = data_full['combined_score'].mean()

# Creating thresholds
fulldata8 = data_full[data_full['combined_score'] > 800]
setsize = int(len(fulldata8))

del fulldata8

all_negative_pairs_prots = pd.read_csv('../DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())

negative_pairs_prots_test = pd.read_csv('../DB/HuriTest/HuRI_negatives.tsv',sep='\t',header=None)
negative_pairs_prots_test = list(tuple(x) for x in negative_pairs_prots_test.to_numpy())

# Pipeline
embedtrainCSV = pd.read_csv('../DB/HuriTest/embeddings_train.csv',index_col=0)
embeddings_train_array = np.array(list(embedtrainCSV.values))
train_embeddings = {embedtrainCSV.index[i]: embeddings_train_array[i] for i in range(len(embeddings_train_array))}

embedtestCSV = pd.read_csv('../DB/HuriTest/embeddings_test.csv',index_col=0)
embeddings_test_array = np.array(list(embedtestCSV.values))
test_embeddings = {embedtestCSV.index[i]: embeddings_test_array[i] for i in range(len(embeddings_test_array))}

vector_size = embedtrainCSV.shape[1]
random.seed(42)

huri_data = pd.read_csv('../DB/HuriTest/HuRI.tsv',sep='\t',header=None,names=['protein1', 'protein2'])

huri_prots = np.loadtxt('../DB/HuriTest/HuRI_proteins.txt',dtype=str)
huri_data = huri_data[huri_data["protein1"].isin(huri_prots) & huri_data["protein2"].isin(huri_prots)].reset_index(drop=True)

pairs_prots_huri = []
for d in huri_data.values:
    pairs_prots_huri.append(('http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + d[0],'http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + d[1], 1))

X_test, y_test = [], []
for prot1, prot2, label in pairs_prots_huri:
    emb_prot1 = test_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = test_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_test.append(hada.tolist()[0])
    y_test.append(1)

for prot1, prot2 in negative_pairs_prots_test:
    prot1 = 'http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + prot1
    prot2 = 'http:/www.ensembl.org/Homo_sapiens/Location/View?r=' + prot2
    emb_prot1 = test_embeddings[prot1].reshape(1, vector_size)
    emb_prot2 = test_embeddings[prot2].reshape(1, vector_size)
    hada = np.multiply(emb_prot1, emb_prot2)
    X_test.append(hada.tolist()[0])
    y_test.append(0)

for t in ['Uniform','Static','Growing']:
    logging.info("Starting training with " + t + " dataset")
    metrics_to_csv = {}
    for i in range(800, -1, -200):
        logging.info("Starting training with threshold: " + str(i))
        if t == 'Static':
            if i == 800:
                data = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
            elif i == 600:
                data = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
            elif i == 400:
                data = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
            elif i == 200:
                data = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
            elif i == 0:
                data = data_full.sample(n=setsize,random_state=42)
        elif t == 'Growing':
            if i == 800:
                data = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
            elif i == 600:
                data = data_full.where(data_full['combined_score']>600).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
            elif i == 400:
                data = data_full.where(data_full['combined_score']>400).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
            elif i == 200:
                data = data_full.where(data_full['combined_score']>200).dropna().reset_index(drop=True).sample(frac=0.1,random_state=42)
            elif i == 0:
                data = data_full.sample(frac=0.1,random_state=42)
        else:
            if i == 800:
                data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
                data = data8
            elif i == 600:
                data6 = data_full.where(data_full['combined_score']>600).where(data_full['combined_score']<=800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
                data = pd.concat([data8,data6])
            elif i == 400:
                data4 = data_full.where(data_full['combined_score']>400).where(data_full['combined_score']<=600).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
                data = pd.concat([data8,data6,data4])
            elif i == 200:
                data2 = data_full.where(data_full['combined_score']>200).where(data_full['combined_score']<=400).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
                data = pd.concat([data8,data6,data4,data2])
            elif i == 0:
                data0 = data_full.where(data_full['combined_score']<=200).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
                data = pd.concat([data8,data6,data4,data2,data0])
                
        pairs_prots = []
        for d in data.values:
            pairs_prots.append(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))
        
        negative_pairs_prots_train = random.sample(all_negative_pairs_prots, len(pairs_prots))

        X_train, y_train = [], []
        for prot1, prot2, label in pairs_prots:
            emb_prot1 = train_embeddings[prot1].reshape(1, vector_size)
            emb_prot2 = train_embeddings[prot2].reshape(1, vector_size)
            hada = np.multiply(emb_prot1, emb_prot2)
            X_train.append(hada.tolist()[0])
            y_train.append(1)

        for prot1, prot2, label in negative_pairs_prots_train:
            emb_prot1 = train_embeddings[prot1].reshape(1, vector_size)
            emb_prot2 = train_embeddings[prot2].reshape(1, vector_size)
            hada = np.multiply(emb_prot1, emb_prot2)
            X_train.append(hada.tolist()[0])
            y_train.append(0)
            
        sample_weight = list(data['combined_score'].values)
        sample_weight.extend([mean for i in range(0, len(pairs_prots))])
        sample_weight = np.array(sample_weight)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for m in ['PU','Normal']:
            if m not in metrics_to_csv:
                metrics_to_csv[m] = {}
            logging.info("Starting training with model: " + m)
            clfs = get_classifiers()
            for c in clfs:
                n = type(c).__name__
                if n not in metrics_to_csv[m]:
                    metrics_to_csv[m][n] = pd.DataFrame(columns=['precision', 'recall', 'WAF','accuracy','AUC'])
                #if m == 'Weighted':
                #    c.fit(X_train, y_train, sample_weight = sample_weight)
                #    y_pred = c.predict(X_test)
                if m == 'PU':
                    pu = ElkanotoPuClassifier(estimator = c, hold_out_ratio = 0.1)
                    y_train_pu = np.where(y_test == 0, -1, y_test)
                    pu.fit(X_test, y_test)
                    y_pred = pu.predict(X_train)
                else:
                    c.fit(X_test, y_test)
                    y_pred = c.predict(X_train)
                prec = precision_score(y_train, y_pred, average='weighted')
                rec = recall_score(y_train, y_pred, average='weighted')
                f1 = f1_score(y_train, y_pred, average='weighted')
                acc = accuracy_score(y_train, y_pred)
                auc = roc_auc_score(y_train, y_pred)
                metrics_to_csv[m][n].loc[i] = [prec, rec, f1, acc, auc]
                if i == 0:
                    Path('../Results/30_04Test/HuriTest/').mkdir(parents=True, exist_ok=True)
                    metrics_to_csv[m][n].to_csv('../Results/30_04Test/HuriTest/'+t+'metrics_'+m+n+'.csv', index=False)
            
        

        
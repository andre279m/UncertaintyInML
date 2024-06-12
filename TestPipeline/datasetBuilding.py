# Imports
import pandas as pd
import numpy as np
import random,logging
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

# Files
#protein_file_path = '../DB/9606.protein.enrichment.terms.v12.0.txt'
protein_full_links_file_path = '../DB/9606.protein.links.detailed.v12.0.txt'

# Creating the Knowledge graph
# prots = pd.read_csv(protein_file_path, sep='\t', header=0)
# prots = prots[prots['term'].str.startswith('GO:')].reset_index(drop=True)
# prots = prots['#string_protein_id'].unique().tolist()


embedCSV = pd.read_csv('../DB/HuriTest/embeddings_train.csv',index_col=0)
prots = embedCSV.index.tolist()
prots = [sample.replace('https://string-db.org/network/', '') for sample in prots]

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)].sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)

mean = data_full['combined_score'].mean()
mean = mean

# Creating thresholds
fulldata8 = data_full[data_full['combined_score'] >= 800]
setsize = int(len(fulldata8))

del fulldata8

all_negative_pairs_prots = pd.read_csv('../DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list((x[0].split('https://string-db.org/network/9606.ENSP')[1],x[1].split('https://string-db.org/network/9606.ENSP')[1],x[2]) for x in all_negative_pairs_prots.to_numpy())

# Pipeline
random.seed(42)

logging.info("Starting Static dataset building ...")

for i in range(800, -1, -200):
    if i == 800:
        data = data_full.where(data_full['combined_score']>=800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
    elif i == 600:
        data = data_full.where(data_full['combined_score']>=600).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
    elif i == 400:
        data = data_full.where(data_full['combined_score']>=400).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
    elif i == 200:
        data = data_full.where(data_full['combined_score']>=200).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
    elif i == 0:
        data = data_full.sample(n=setsize,random_state=42)

    # Creating the dataset
    pairs_prots = []
    for d in data.values:
        pairs_prots.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1], d[-1]))

    negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))

    X, y, sample_weight = [], [], []
    for prot1, prot2, c in pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(1)
        sample_weight.append(c)

    for prot1, prot2, label in negative_pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(0)
        sample_weight.append(mean)

    # Creating training set and test set
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    X, y ,sample_weight = np.array(X), np.array(y), np.array(sample_weight)
    Path('../DB/Static/T'+str(i)+'/').mkdir(parents=True, exist_ok=True)
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sample_weight_train = sample_weight[train_index]
        logging.info("Writing dataset with threshold: " + str(i) + " and fold: " + str(j))
        np.savetxt('../DB/Static/T'+str(i)+'/X_train_Fold'+str(j)+'.csv', X_train, delimiter=',',fmt='%s')
        np.savetxt('../DB/Static/T'+str(i)+'/X_test_Fold'+str(j)+'.csv', X_test, delimiter=',',fmt='%s')
        np.savetxt('../DB/Static/T'+str(i)+'/y_train_Fold'+str(j)+'.csv', y_train, delimiter=',')
        np.savetxt('../DB/Static/T'+str(i)+'/y_test_Fold'+str(j)+'.csv', y_test, delimiter=',')
        np.savetxt('../DB/Static/T'+str(i)+'/sample_weight_train_Fold'+str(j)+'.csv', sample_weight_train, delimiter=',')

logging.info("Starting Growing dataset building...")

for i in range(800, -1, -200):
    if i == 800:
        data = data_full.where(data_full['combined_score']>=800).dropna().reset_index(drop=True).sample(frac=0.2,random_state=42)
    elif i == 600:
        data = data_full.where(data_full['combined_score']>=600).dropna().reset_index(drop=True).sample(frac=0.2,random_state=42)
    elif i == 400:
        data = data_full.where(data_full['combined_score']>=400).dropna().reset_index(drop=True).sample(frac=0.2,random_state=42)
    elif i == 200:
        data = data_full.where(data_full['combined_score']>=200).dropna().reset_index(drop=True).sample(frac=0.2,random_state=42)
    elif i == 0:
        data = data_full.sample(frac=0.2,random_state=42)

    # Creating the dataset
    pairs_prots = []
    for d in data.values:
        pairs_prots.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1], d[-1]))

    negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))

    X, y, sample_weight = [], [], []
    for prot1, prot2, c in pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(1)
        sample_weight.append(c)

    for prot1, prot2, label in negative_pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(0)
        sample_weight.append(mean)

    # Creating training set and test set
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    X, y ,sample_weight = np.array(X), np.array(y), np.array(sample_weight)
    Path('../DB/Growing/T'+str(i)+'/').mkdir(parents=True, exist_ok=True)
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sample_weight_train = sample_weight[train_index]
        logging.info("Writing dataset with threshold: " + str(i) + " and fold: " + str(j))
        np.savetxt('../DB/Growing/T'+str(i)+'/X_train_Fold'+str(j)+'.csv', X_train, delimiter=',',fmt='%s')
        np.savetxt('../DB/Growing/T'+str(i)+'/X_test_Fold'+str(j)+'.csv', X_test, delimiter=',',fmt='%s')
        np.savetxt('../DB/Growing/T'+str(i)+'/y_train_Fold'+str(j)+'.csv', y_train, delimiter=',')
        np.savetxt('../DB/Growing/T'+str(i)+'/y_test_Fold'+str(j)+'.csv', y_test, delimiter=',')
        np.savetxt('../DB/Growing/T'+str(i)+'/sample_weight_train_Fold'+str(j)+'.csv', sample_weight_train, delimiter=',')

logging.info("Starting Uniform dataset building...")

for i in range(800, -1, -200):
    if i == 800:
        data8 = data_full.where(data_full['combined_score']>=800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
        data = data8
    elif i == 600:
        data6 = data_full.where(data_full['combined_score']>=600).where(data_full['combined_score']<800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
        data = pd.concat([data8,data6])
    elif i == 400:
        data4 = data_full.where(data_full['combined_score']>=400).where(data_full['combined_score']<600).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
        data = pd.concat([data8,data6,data4])
    elif i == 200:
        data2 = data_full.where(data_full['combined_score']>=200).where(data_full['combined_score']<400).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
        data = pd.concat([data8,data6,data4,data2])
    elif i == 0:
        data0 = data_full.where(data_full['combined_score']<200).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
        data = pd.concat([data8,data6,data4,data2,data0])

    # Creating the dataset
    pairs_prots = []
    for d in data.values:
        pairs_prots.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1], d[-1]))

    negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))

    X, y, sample_weight = [], [], []
    for prot1, prot2, c in pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(1)
        sample_weight.append(c)

    for prot1, prot2, label in negative_pairs_prots:
        line = [prot1,prot2]
        X.append(line)
        y.append(0)
        sample_weight.append(mean)

    # Creating training set and test set
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    X, y ,sample_weight = np.array(X), np.array(y), np.array(sample_weight)
    Path('../DB/Uniform/T'+str(i)+'/').mkdir(parents=True, exist_ok=True)
    for j, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sample_weight_train = sample_weight[train_index]
        logging.info("Writing dataset with threshold: " + str(i) + " and fold: " + str(j))
        np.savetxt('../DB/Uniform/T'+str(i)+'/X_train_Fold'+str(j)+'.csv', X_train, delimiter=',',fmt='%s')
        np.savetxt('../DB/Uniform/T'+str(i)+'/X_test_Fold'+str(j)+'.csv', X_test, delimiter=',',fmt='%s')
        np.savetxt('../DB/Uniform/T'+str(i)+'/y_train_Fold'+str(j)+'.csv', y_train, delimiter=',')
        np.savetxt('../DB/Uniform/T'+str(i)+'/y_test_Fold'+str(j)+'.csv', y_test, delimiter=',')
        np.savetxt('../DB/Uniform/T'+str(i)+'/sample_weight_train_Fold'+str(j)+'.csv', sample_weight_train, delimiter=',')


# logging.info("Starting UnderSampled dataset building...")

# for i in range(800, -1, -200):
#     if i == 800:
#         data8 = data_full.where(data_full['combined_score']>800).dropna().reset_index(drop=True).sample(n=setsize,random_state=42)
#         data = data8
#     elif i == 600:
#         data6 = data_full.where(data_full['combined_score']>600).where(data_full['combined_score']<=800).dropna().reset_index(drop=True).sample(n=int(setsize*(8/10)),random_state=42)
#         data = pd.concat([data8,data6])
#     elif i == 400:
#         data4 = data_full.where(data_full['combined_score']>400).where(data_full['combined_score']<=600).dropna().reset_index(drop=True).sample(n=int(setsize*(6/10)),random_state=42)
#         data = pd.concat([data8,data6,data4])
#     elif i == 200:
#         data2 = data_full.where(data_full['combined_score']>200).where(data_full['combined_score']<=400).dropna().reset_index(drop=True).sample(n=int(setsize*(4/10)),random_state=42)
#         data = pd.concat([data8,data6,data4,data2])
#     elif i == 0:
#         data0 = data_full.where(data_full['combined_score']<=200).dropna().reset_index(drop=True).sample(n=int(setsize*(2/10)),random_state=42)
#         data = pd.concat([data8,data6,data4,data2,data0])

#     # Creating the dataset
#     pairs_prots = []
#     for d in data.values:
#         pairs_prots.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1], d[-1]))

#     negative_pairs_prots = random.sample(all_negative_pairs_prots, len(pairs_prots))

#     X, y, sample_weight = [], [], []
#     for prot1, prot2, c in pairs_prots:
#         line = [prot1,prot2]
#         X.append(line)
#         y.append(1)
#         sample_weight.append(c)

#     for prot1, prot2, label in negative_pairs_prots:
#         line = [prot1,prot2]
#         X.append(line)
#         y.append(0)
#         sample_weight.append(mean)

#     # Creating training set and test set
#     skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
#     X, y ,sample_weight = np.array(X), np.array(y), np.array(sample_weight)
#     Path('../DB/Undersampling/T'+str(i)+'/').mkdir(parents=True, exist_ok=True)
#     for j, (train_index, test_index) in enumerate(skf.split(X, y)):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         sample_weight_train = sample_weight[train_index]
#         logging.info("Writing dataset with threshold: " + str(i) + " and fold: " + str(j))
#         np.savetxt('../DB/Undersampling/T'+str(i)+'/X_train_Fold'+str(j)+'.csv', X_train, delimiter=',',fmt='%s')
#         np.savetxt('../DB/Undersampling/T'+str(i)+'/X_test_Fold'+str(j)+'.csv', X_test, delimiter=',',fmt='%s')
#         np.savetxt('../DB/Undersampling/T'+str(i)+'/y_train_Fold'+str(j)+'.csv', y_train, delimiter=',')
#         np.savetxt('../DB/Undersampling/T'+str(i)+'/y_test_Fold'+str(j)+'.csv', y_test, delimiter=',')
#         np.savetxt('../DB/Undersampling/T'+str(i)+'/sample_weight_train_Fold'+str(j)+'.csv', sample_weight_train, delimiter=',')

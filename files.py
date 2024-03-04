# Imports
import pandas as pd
import numpy as np
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

# Creating thresholds
fulldata8 = data_full[data_full['combined_score'] > 800]
setsize = int(len(fulldata8)/2)
testsize = int(len(data_full)*0.1)
select = [int((N*len(fulldata8))/setsize) for N in range(setsize)]
train_data8 = fulldata8.loc[select]
train_data8.to_csv('DB/TRAIN/train_data8.csv', index=False)

fulldata6 = data_full[data_full['combined_score'] > 600]
select = [int((N*len(fulldata6))/setsize) for N in range(setsize)]
train_data6 = fulldata6.loc[select]
train_data6.to_csv('DB/TRAIN/train_data6.csv', index=False)

fulldata4 = data_full[data_full['combined_score'] > 400]
select = [int((N*len(fulldata4))/setsize) for N in range(setsize)]
train_data4 = fulldata4.loc[select]
train_data4.to_csv('DB/TRAIN/train_data4.csv', index=False)

fulldata2 = data_full[data_full['combined_score'] > 200]
select = [int((N*len(fulldata2))/setsize) for N in range(setsize)]
train_data2 = fulldata2.loc[select]
train_data2.to_csv('DB/TRAIN/train_data2.csv', index=False)

select = [int((N*len(data_full))/setsize) for N in range(setsize)]
train_data0 = data_full.loc[select]
train_data0.to_csv('DB/TRAIN/train_data0.csv', index=False)

dt_test = data_full[~data_full.isin(train_data8)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)
dt_test = dt_test[~dt_test.isin(train_data6)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)
dt_test = dt_test[~dt_test.isin(train_data4)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)
dt_test = dt_test[~dt_test.isin(train_data2)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)
dt_test = dt_test[~dt_test.isin(train_data0)].dropna().sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)

for i in range(5):
    select = [int(((N*len(dt_test))/testsize) + 1) for N in range(testsize)]
    test_data = dt_test.loc[select]
    test_data.to_csv('DB/TEST/test_data'+str(i)+'.csv', index=False)

all_negative_pairs_prots = pd.read_csv('DB/negative_pairs_prots.csv', header=0)
all_negative_pairs_prots = list(tuple(x) for x in all_negative_pairs_prots.to_numpy())
all_negative_pairs_prots = pd.DataFrame(all_negative_pairs_prots, columns=['protein1', 'protein2','label'])

negative_train8 = all_negative_pairs_prots.sample(len(train_data8))
negative_train6 = all_negative_pairs_prots.sample(len(train_data6))
negative_train4 = all_negative_pairs_prots.sample(len(train_data4))
negative_train2 = all_negative_pairs_prots.sample(len(train_data2))
negative_train0 = all_negative_pairs_prots.sample(len(train_data0))
all_negative_pairs_prots = all_negative_pairs_prots[~all_negative_pairs_prots.isin(negative_train8)].dropna().reset_index(drop=True)
all_negative_pairs_prots = all_negative_pairs_prots[~all_negative_pairs_prots.isin(negative_train6)].dropna().reset_index(drop=True)
all_negative_pairs_prots = all_negative_pairs_prots[~all_negative_pairs_prots.isin(negative_train4)].dropna().reset_index(drop=True)
all_negative_pairs_prots = all_negative_pairs_prots[~all_negative_pairs_prots.isin(negative_train2)].dropna().reset_index(drop=True)
all_negative_pairs_prots = all_negative_pairs_prots[~all_negative_pairs_prots.isin(negative_train0)].dropna().reset_index(drop=True)

negative_train8.to_csv('DB/TRAIN/negative_train8.csv', index=False)
negative_train6.to_csv('DB/TRAIN/negative_train6.csv', index=False)
negative_train4.to_csv('DB/TRAIN/negative_train4.csv', index=False)
negative_train2.to_csv('DB/TRAIN/negative_train2.csv', index=False)
negative_train0.to_csv('DB/TRAIN/negative_train0.csv', index=False)

for i in range(5):
    select = [int(((N*len(all_negative_pairs_prots))+1)/testsize) for N in range(testsize)]
    negative_pairs_prots = all_negative_pairs_prots.loc[select]
    negative_pairs_prots.to_csv('DB/TEST/negative_pairs_prots'+str(i)+'.csv', index=False)

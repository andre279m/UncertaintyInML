import pandas as pd

# Path to the file with the full STRING dataset
protein_full_links_file_path = '../DB/9606.protein.links.full.v11.0.txt'

embedCSV = pd.read_csv('../DB/HuriTest/embeddings_train.csv',index_col=0)
prots = embedCSV.index.tolist()
prots = [sample.replace('https://string-db.org/network/', '') for sample in prots]

# Distribution of Confidence
data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)].sort_values(by=['combined_score'], ascending=False).reset_index(drop=True)

all_negative_pairs_prots = []
# Without semantic similarity
pairs_prots_STRING = set()
for d in data_full.values:
    pairs_prots_STRING.add(('https://string-db.org/network/' + d[0],'https://string-db.org/network/' + d[1], 1))
# Creating a list of all proteins
prots_links = ['https://string-db.org/network/' + prot for prot in prots]
for prot in prots_links:
    for prot2 in prots_links:
        if (prot,prot2,1) in pairs_prots_STRING or (prot2,prot,1) in pairs_prots_STRING:
            continue
        elif prot != prot2:
            all_negative_pairs_prots.append((prot,prot2,0))

all_negative_pairs_prots = list(set(all_negative_pairs_prots))

all_negative_pairs_prots = pd.DataFrame(all_negative_pairs_prots, columns=['protein1', 'protein2', 'interaction'])
STRING_negative_pairs_prots = all_negative_pairs_prots.sample(n = len(data_full), random_state=42)
STRING_negative_pairs_prots.to_csv('../DB/negative_pairs_prots.csv', index=False)
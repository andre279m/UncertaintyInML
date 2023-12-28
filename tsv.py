import pandas as pd

protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
protein_full_links_file_path = 'DB/9606.protein.links.detailed.v12.0.txt'

prots = []

string_go_terms = {}

with open(protein_file_path , 'r') as prot_annot:
    prot_annot.readline()
    for line in prot_annot:
        elements_annot = line.split('\t')
        id_prot, GO_term = elements_annot[0], elements_annot[2]
        if GO_term.startswith('GO:') :
            url_GO_term = 'http://purl.obolibrary.org/obo/GO_' + GO_term.split(':')[1]

            if id_prot not in prots:
                prots.append(id_prot)

            id_prot = id_prot.split('9606.ENSP')[1]
            if id_prot in string_go_terms:
                string_go_terms[id_prot] += ';' + url_GO_term
            else:
                string_go_terms[id_prot] = url_GO_term

annotations = pd.DataFrame.from_dict(string_go_terms, orient='index')
annotations.to_csv('DB/SSMC-master/annotations_STRING_GO.tsv', sep='\t', header=False)

data_full = pd.read_csv(protein_full_links_file_path, sep=" ", header=0)
data_full = data_full[data_full["protein1"].isin(prots) & data_full["protein2"].isin(prots)]

pairs_prots_STRING = []
for d in data_full.values:
    pairs_prots_STRING.append((d[0].split('9606.ENSP')[1],d[1].split('9606.ENSP')[1]))

pairs_prots_STRING = set(pairs_prots_STRING)

# Creating a list of all proteins
prots_links = [prot.split('9606.ENSP')[1] for prot in prots]

negative_pairs_prots = []
for prot in prots_links:
    for prot2 in prots_links:
        if (prot,prot2) in pairs_prots_STRING:
            continue
        elif prot != prot2:
            negative_pairs_prots.append((prot,prot2))

negative_pairs_prots = pd.DataFrame(negative_pairs_prots)
size=int(len(pairs_prots_STRING)*0.1)
print(size)
negative_pairs_prots = negative_pairs_prots.sample(n=size, random_state=42)
negative_pairs_prots.to_csv('DB/SSMC-master/negative_pairs_prots.tsv', sep='\t', header=False,index=False)
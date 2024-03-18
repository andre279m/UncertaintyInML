import rdflib
from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import UniformSampler, ObjFreqSampler, PredFreqSampler
from pyrdf2vec.walkers import RandomWalker, WalkletWalker
from pyrdf2vec.walkers.weisfeiler_lehman import WLWalker
import pandas as pd

gene_ontology_file_path = 'DB/go.owl'
protein_file_path = 'DB/9606.protein.enrichment.terms.v12.0.txt'
gene_ontology_annotated_file_path = 'DB/go_annotated.owl'

g = rdflib.Graph()
g.parse(gene_ontology_file_path, format = 'xml');

prots_train = []

with open(protein_file_path , 'r') as prot_annot:
    prot_annot.readline()
    for line in prot_annot:
        elements_annot = line.split('\t')
        id_prot, GO_term = elements_annot[0], elements_annot[2]
        if GO_term.startswith('GO:') :
            url_GO_term = 'http://purl.obolibrary.org/obo/GO_' + GO_term.split(':')[1]
            url_prot = 'https://string-db.org/network/' + id_prot
            if id_prot not in prots_train:
                prots_train.append(id_prot)
            g.add((rdflib.term.URIRef(url_prot), rdflib.term.URIRef('http://purl.obolibrary.org/obo/go.owl#has_function') , rdflib.term.URIRef(url_GO_term)))


# TODO: Add the protein links to the graph
huri_pairs = pd.read_csv('DB/HuriTest/HuRI.tsv', sep = ' ')
mappings = pd.read_csv('DB/HuriTest/idmapping_2024_03_15.tsv', sep = ' ')
goa = pd.read_csv('DB/HuriTest/goa_human.gaf', sep = '\t', comment = '!', header = None)



# Defining rdf2vec paramenters
vector_size = 200
n_walks = 100
type_word2vec = 'skip-gram'
walk_depth = 4
walker_type = 'wl'
sampler_type = 'uniform'

# Creating a pyrdf2vec graph
# Creating a pyrdf2vec graph
# g_pyrdf2vec = kg.KG(mul_req=False)
# for (s, p, o) in g:
#     s_v = Vertex(str(s))
#     o_v = Vertex(str(o))
#     p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
#     g_pyrdf2vec.add_vertex(s_v)
#     g_pyrdf2vec.add_vertex(p_v)
#     g_pyrdf2vec.add_vertex(o_v)
#     g_pyrdf2vec.add_edge(s_v, p_v)
#     g_pyrdf2vec.add_edge(p_v, o_v)

g_pyrdf2vec = kg.KG("DB/go_annotated2.owl", mul_req=False)

# Defining the word2vec strategy
if type_word2vec == 'CBOW':
    sg_value = 0
elif type_word2vec == 'skip-gram':
    sg_value = 1

# Defining sampling strategy
if sampler_type.lower() == 'uniform':
    sampler = UniformSampler()
elif sampler_type.lower() == 'predfreq':
    sampler = PredFreqSampler()
elif sampler_type.lower() == 'objfreq':
    sampler = ObjFreqSampler()

# Defining walker strategy
if walker_type.lower() == 'random':
    walker = RandomWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler, n_jobs = -1)
elif walker_type.lower() == 'wl':
#    walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler, n_jobs = -1)
    walker = WLWalker(max_depth=walk_depth, max_walks=n_walks, sampler = sampler, n_jobs = -1)
elif walker_type.lower() == 'walklet':
    walker = WalkletWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler, n_jobs = -1 )

# testing RDF2Vec embeddings
transformer = RDF2VecTransformer(Word2Vec(size=vector_size, sg=sg_value), walkers=[walker])

all_prots = prots_train + prots_test

# Generating the embeddings
embeddings = transformer.fit_transform(g_pyrdf2vec, all_prots)
dict_embeddings_train = {prots_train[i]: embeddings[i] for i in range(len(prots_train))}
dict_embeddings_test = {prots_test[i]: embeddings[i] for i in range(len(prots_test))}
dict_embeddings.to_csv('DB/embeddings.csv', index=True)
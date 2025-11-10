import spacy

import pandas as pd

from Bio import Entrez, Medline

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import sacremoses

from transformers import pipeline, set_seed

from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModel

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import itertools

import os

import Levenshtein

from Levenshtein import distance

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'and', 'but', 'if', 'or',
    'as', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'it',
    'its', 'itself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they',
    'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am',

    # Academic & Scientific Jargon
    'abstract', 'introduction', 'background', 'methods', 'method', 'methodology',
    'results', 'result', 'conclusion', 'conclusions', 'discussion', 'summary',
    'study', 'studies', 'research', 'analysis', 'analyses', 'data', 'dataset',
    'datasets', 'evidence', 'findings', 'finding', 'implications', 'implication',
    'application', 'applications', 'purpose', 'objective', 'objectives', 'aim',
    'aims', 'scope', 'limitation', 'limitations', 'future work', 'acknowledgments',
    'references', 'figure', 'figures', 'fig', 'table', 'tables', 'author', 'authors',
    'et al', 'publication', 'paper', 'article', 'review', 'case', 'report',
    'case report', 'case series', 'cohort', 'manuscript',

    # Common Verbs
    'using', 'showed', 'found', 'demonstrated', 'observed', 'investigated',
    'described', 'reported', 'suggested', 'analyzed', 'compared', 'identified',
    'characterized', 'evaluated', 'developed', 'performed', 'confirmed',
    'increase', 'decrease', 'increasing', 'decreasing', 'based', 'associated',
    'related', 'involved', 'including', 'following', 'comprising',

    # Common Adjectives/Adverbs
    'significant', 'significantly', 'novel', 'new', 'respective', 'respectively',
    'various', 'multiple', 'several', 'diverse', 'certain', 'specific',
    'different', 'potential', 'further', 'additional', 'recent', 'recently',
    'high', 'low', 'higher', 'lower', 'large', 'small', 'major', 'minor',
    'important', 'key', 'pivotal', 'crucial', 'main', 'primary', 'secondary',
    'good', 'better', 'best', 'well', 'poor', 'common', 'rare', 'current',
    'first', 'second', 'third', 'i', 'ii', 'iii', 'iv', 'v', 'c', 'e', 'g',

    # Generic Biomedical / Scientific Concepts
    'role', 'roles', 'impact', 'impacts', 'effect', 'effects', 'function', 'functions',
    'mechanism', 'mechanisms', 'pathway', 'pathways', 'process', 'processes',
    'system', 'systems', 'complex', 'complexes', 'factor', 'factors', 'activity',
    'activities', 'level', 'levels', 'patient', 'patients', 'individual', 'individuals',
    'human', 'humans', 'animal', 'animals', 'mouse', 'mice', 'rat', 'rats',
    'group', 'groups', 'sample', 'samples', 'tissue', 'tissues', 'cell', 'cells',
    'expression', 'regulation', 'upregulation', 'downregulation', 'production',
    'response', 'responses', 'association', 'correlation', 'relationship',
    'interaction', 'interactions', 'development', 'progression', 'proliferation',
    'metastasis', 'disease', 'diseases', 'disorder', 'disorders', 'syndrome',
    'condition', 'cancer', 'cancers', 'tumor', 'tumors', 'therapy', 'therapies',
    'treatment', 'treatments', 'therapeutic', 'drug', 'drugs', 'target', 'targets',
    'molecule', 'molecules', 'protein', 'proteins', 'gene', 'genes', 'rna', 'dna',
    'noncoding rna', 'noncoding rnas', 'coding', 'non-coding', 'region', 'regions',
    'domain', 'domains', 'family', 'families', 'type', 'types', 'class', 'classification',
    'number', 'values', 'value', 'field', 'area', 'information', 'knowledge',
    'understanding', 'approach', 'strategy', 'strategies', 'tool', 'tools',
    'platform', 'technology', 'characteristic', 'characteristics', 'feature', 'features',
    'property', 'properties', 'context', 'context', 'issue', 'issues', 'challenge',
    'challenges', 'need', 'presence', 'absence', 'change', 'changes', 'percent',
    'rate', 'rates', 'risk', 'risks', 'factor', 'factors', 'a key', 'a pivotal role',
    'a major', 'a significant', 'a combination', 'our understanding', 'a new',
    'these findings', 'our findings', 'this study', 'recent studies', 'a leading cause',
    'growing evidence', 'recent research',
}


device = torch.device("cpu")

BERT_model = 'dmis-lab/biobert-base-cased-v1.1'

biobert_tokenizer = AutoTokenizer.from_pretrained(BERT_model)

biobert_model = biobert_model = AutoModel.from_pretrained(BERT_model)

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

tokenizer =  BioGptTokenizer.from_pretrained("microsoft/biogpt")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer,device = device)

set_seed(42)

bio_summary = ""

all_entities = []

unique_entities = []

use_pairs = []

threshold = 0.90

nlp = spacy.load("en_ner_bionlp13cg_md")
topic = input("What field do you want to discover novel research opportunities in?")
Entrez.email = "matthewoleynikov1@gmail.com"

handle = Entrez.esearch(db="pubmed", term = topic, retmax = 30)
record = Entrez.read(handle)
idlist = record["IdList"]
print(idlist)
for x in idlist:
    portal = Entrez.efetch(db = "pubmed" , id = x, rettype = "medline", retmode = "text")
    for record in Medline.parse(portal):
        title = record.get("TI","Untitled")
        
        abstract = record.get("AB","No Summary")
        bio_summary += f"{title}\n{abstract}\n\n"
        doc = nlp(abstract)
        for ent in doc.ents:
                all_entities.append(ent.text)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ["NOUN","PROPN","ADJ"] and len(chunk.text) > 2:   
                all_entities.append(chunk.text)
        #print(abstract)
#print(all_entities)

for ent in all_entities:
    if ent not in unique_entities:
        unique_entities.append(ent)
print("The unique entities")
#print(unique_entities)
#BioBERT***************************************

def get_embedding(ent):
    if not ent or ent.isspace():
        return np.zeros(768)
    inputs = biobert_tokenizer(ent,return_tensors = "pt", truncation = True, padding = True, max_length = 512)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
    if outputs.last_hidden_state is None or outputs.last_hidden_state.shape[1] == 0:
        return np.zeros(768)
    else:
        cls_tensor = outputs.last_hidden_state[0,0,:]
        return cls_tensor.cpu().numpy()

#Semantic analysis*****************************

print("Generating embeddings")
ent_embedding_list = []
valid_ents = []
for ent in unique_entities:
    embedding = get_embedding(ent)
    if np.any(embedding):
        ent_embedding_list.append(embedding)
        valid_ents.append(ent)

if not ent_embedding_list:
    print("no valid embeddings")
    exit()

embedding_array = np.array(ent_embedding_list)

print("\ngenerating matrix of similarity(may take a while)")
similarity_matrix = cosine_similarity(embedding_array)
#print(similarity_matrix)
for i, j in itertools.combinations(range(len(valid_ents)),2):
   score = similarity_matrix[i,j]
   if Levenshtein.distance(valid_ents[i],valid_ents[j]) > 2:

    if valid_ents[i].lower() not in STOP_WORDS and valid_ents[j].lower() not in STOP_WORDS:
        if valid_ents[i].lower() != valid_ents[j].lower() and valid_ents[i].lower() not in valid_ents[j].lower() and valid_ents[j].lower() not in valid_ents[i].lower():
            if score >= threshold and score < 0.999:
                pair_data = {
                "entity_1" : valid_ents[i],
                "entity_2" : valid_ents[j],
                "score" : score
                }
                use_pairs.append(pair_data)
print(use_pairs)
#BioGPT****************************************

input_text = f"""
Based on the following biomedical information:
{bio_summary}

Novel RNA splicing research could investigate: 

"""
#2. Suggest a novel research hypothesis related to {topic}.
#3. Explain the reasoning behind the hypothesis.
#response = generator(input_text,max_new_tokens = 1024, num_return_sequences = 1, do_sample = True)
#text = response[0]["generated_text"]

#print(text[len(bio_summary)+46:])   
print("done")
#use_pairs.sort(key=lambda x: x["score"], reverse=True)
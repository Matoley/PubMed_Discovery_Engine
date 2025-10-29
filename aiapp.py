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

BERT_model = 'dmis-lab/biobert-base-cased-v1.1'

biobert_tokenizer = AutoTokenizer.from_pretrained(BERT_model)

biobert_model = biobert_model = AutoModel.from_pretrained(BERT_model)

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

tokenizer =  BioGptTokenizer.from_pretrained("microsoft/biogpt")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer)

set_seed(42)

bio_summary = ""

all_entities = []

unique_entities = []



nlp = spacy.load("en_core_sci_lg")
topic = input("What field do you want to discover novel research opportunities in?")
Entrez.email = "matthewoleynikov1@gmail.com"

handle = Entrez.esearch(db="pubmed", term = topic, retmax = 3)
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
print(unique_entities)
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
print(similarity_matrix)
#for i, j in itertools.combinations(range(len(unique_entities)),2):
   
#BioGPT****************************************

input_text = f"""
Based on the following biomedical information:
{bio_summary}

Novel RNA splicing research could investigate: 

"""
#2. Suggest a novel research hypothesis related to {topic}.
#3. Explain the reasoning behind the hypothesis.
response = generator(input_text,max_new_tokens = 1024, num_return_sequences = 1, do_sample = True)
text = response[0]["generated_text"]

#print(text[len(bio_summary)+46:])   
print("done")
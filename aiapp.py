from key_words import STOP_WORDS, SIGNIFICANT_LABELS

import spacy

from Bio import Entrez, Medline

import torch

from transformers import pipeline, set_seed

from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModel

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import itertools

import os

import Levenshtein

from Levenshtein import distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_model = 'dmis-lab/biobert-base-cased-v1.1'

biobert_tokenizer = AutoTokenizer.from_pretrained(BERT_model)

biobert_model = AutoModel.from_pretrained(BERT_model)

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

tokenizer =  BioGptTokenizer.from_pretrained("microsoft/biogpt")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer,device = device, return_full_text=False)

set_seed(42)

bio_summary = ""

all_entities = []

unique_entities = []

use_pairs = []

top_threshold = 0.95

bottom_threshold = 0.85

nlp = spacy.load("en_ner_bionlp13cg_md")
topic = input("What field do you want to discover novel research opportunities in?")
Entrez.email = os.getenv("PUBMED_EMAIL")

handle = Entrez.esearch(db="pubmed", term = topic, retmax = 100)
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
            if ent.label_ in SIGNIFICANT_LABELS:
                all_entities.append(ent.text)
        #for chunk in doc.noun_chunks:
            #if chunk.root.pos_ in ["NOUN","PROPN","ADJ"] and len(chunk.text) > 2:   
                #all_entities.append(chunk.text)
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
            if score >= bottom_threshold and score <= top_threshold and score < 0.999:
                pair_data = {
                "entity_1" : valid_ents[i],
                "entity_2" : valid_ents[j],
                "score" : score
                }
                use_pairs.append(pair_data)
#print(use_pairs)

if not use_pairs:
    print(f"No novel connections found in the array ({bottom_threshold}-{top_threshold}).")
    print("Try broadening your topic or increasing the RETMAX value. Exiting.")
    exit()
#BioGPT****************************************
use_pairs.sort(key=lambda x: x["score"], reverse=True)
top_10_pairs = use_pairs[:10]
print(top_10_pairs)
for pairs in top_10_pairs:
    ent_A = pairs["entity_1"]
    ent_B = pairs["entity_2"]
    score = pairs["score"]
    input_text = f"A novel research hypothesis linking {ent_A} and {ent_B} in the context of {topic} is that"

    response = generator(input_text,max_new_tokens = 1024, num_return_sequences = 1, do_sample = True)
    clean_text = response[0]["generated_text"].strip()
    if clean_text:
            print(clean_text)
    else:
        print("BioGPT returned an empty string. The prompt might be too complex for this pair.")
    print("\n ********************************************************************************")
print("done")

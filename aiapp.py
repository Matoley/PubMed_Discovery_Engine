import spacy

import pandas as pd

from Bio import Entrez, Medline

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import sacremoses

from transformers import pipeline, set_seed

from transformers import BioGptTokenizer, BioGptForCausalLM





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

handle = Entrez.esearch(db="pubmed", term = topic, retmax = 10)
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
        print(abstract)
print(all_entities)

for ent in all_entities:
    if ent not in unique_entities:
        unique_entities.append(ent)
print("The unique entities")
print(unique_entities)
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

print(text[len(bio_summary)+46:])   
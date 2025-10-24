# PubMed_Discovery_Engine
Open-source AI tool using BioBert and BioGPT to generate novel research hypothesis

The goal of the project is to assist in biomedical research:
  Biomedical research often requires perusing vast amount of medical publication. The sheer volume of publications means that advancement in innovation is hindered by finding connections and typically, advanced text-mining tools are inaccessible to many or require formidable amounts    of experience with coding.

****

Our Solution: ****An AI Python Pipeline****
The project is planned to work in 4 steps
1. Fetches data from PubMed Abstracts from a user's input

2. ****Extracting key words****:
   Utilizing a special biomedical NLP (scispaCy), key entities are found in the abstracts and compliled into a list (gene names, chemials, effects, etc)

3. ****Finding hidden correlations****:
   The program uses BioBERT to find semantic relationships between key words from abstracts. This step is crucial to find potential latent relationships based on context that led to novel discovery.

4. ****Generating Novel Hypothesis****
   The program feeds the most promising correlations based off BioBERT vectors (embeddings) to a specially biomedical tuned LLM (BioGPT). The model is prompted to provided plausible hypothesis that relate to the given entities.
     
****What is completed/being worked on?****
1. PubMed (Done)
2. scispaCy (Near done)
3. BioBERT (Starting soon)
4. BioGPT (Currently done, but integration with BioBERT may change status)


# PubMed_Discovery_Engine
Open-source AI tool using BioBert and BioGPT to generate novel research hypothesis

The goal of the project is to assist in biomedical research:
  Biomedical research often requires perusing vast amounts of medical publication. The sheer volume of publications means that advancement in innovation is hindered by finding connections and typically, advanced text-mining tools are inaccessible to many or require formidable amounts    of experience with coding.

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

This project utilizes the following open-source frameworks and models:

* **scispaCy**
    Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing. [Link to Paper](https://doi.org/10.18653/v1/W19-5034)

* **BioBERT**
    Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. [Link to Paper](https://doi.org/10.1093/bioinformatics/btz682)

* **BioGPT**
    Luo, R., et al. (2022). BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining. [Link to Paper](https://doi.org/10.1093/bib/bbac409)

Thank you to **Dr. Alexander Varshavsky** for his guidance and feedback that assisted in evaluating the uses of this tool

****
     
## Set up

Follow these steps to run a local copy of the program

### Prerequisites

Python 3.8+ and pip must be installed, along with other program requirements (see requirements.txt)

### Installation

1.  Clone the repo:
    ```sh
    git clone [https://github.com/Matoley/PubMed_Discovery_Engine.git](https://github.com/Matoley/PubMed_Discovery_Engine.git)
    ```
2.  Navigate into the project directory:
    ```sh
    cd PubMed_Discovery_Engine
    ```
3.  Create and activate a virtual environment:
    ```sh
    # On macOS/Linux
    python3 -m venv venv-stable
    source venv-stable/bin/activate
    
    # On Windows
    python -m venv venv-stable
    .\venv-stable\Scripts\activate
    ```
4.  Install all the required packages:
    ```sh
    pip install -r requirements.txt
    ```
After processing 100 abstracts, it will output the top 10 results.

### Run the app

Simply run the main Python script:
```bash
python aiapp.py
```
It will ask you for a topic. Give an example, preferably something specific (Ubiquitination, Circadian Clock, etc.) and press Enter.
```
What field do you want to discover novel research opportunities in?
```


# Retrofitting Word Vectors with Semantic Resources

This project reimplements the retrofitting method proposed by Faruqui et al. (2015), *Retrofitting Word Vectors to Semantic Lexicons*. The goal is to improve static word embeddings by injecting information from semantic lexical resources such as WordNet, PPDB, and WOLF.

The main idea is that a retrofitted word vector should stay close to its original pretrained vector while also becoming closer to semantically related words in a lexical graph.

## Method

Given an original embedding matrix and a semantic graph, each word vector is updated iteratively according to:

```text
q_i = (alpha_i * qhat_i + sum_j beta_ij * q_j)
      / (alpha_i + sum_j beta_ij)
```

where:

- `qhat_i` is the original pretrained vector of word `i`;
- `q_i` is the retrofitted vector;
- `q_j` are the vectors of semantic neighbors of word `i`;
- `alpha_i` controls how strongly the model keeps the original vector;
- `beta_ij` controls how strongly the model trusts each semantic neighbor.

The implementation uses synchronous iterative updates: at each iteration, all new vectors are computed from the vectors of the previous iteration. This optimizes the same retrofitting objective while making the update process deterministic and easy to inspect.

## Project Structure

```text
retrofitting-wordvectors-final/
├── README.md
├── src/                         # core algorithm and alpha search scripts
│   ├── retrofit.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── run_english_alpha_search.py
│   └── run_french_alpha_search.py
├── resources_prepa/             # preparation scripts for semantic resources and embeddings
│   ├── prepare_wordnet.py
│   ├── prepare_PPDB.py
│   ├── prepare_french.py
│   ├── prepare_word2vec.py
│   ├── download_word2vec.py
│   └── download_french_resources.py
├── evaluation/                  # evaluation scripts for similarity and sentiment tasks
│   ├── eval_utils.py
│   ├── run_glove_word_similarity.py
│   ├── run_word2vec_word_similarity.py
│   ├── run_french_word_similarity.py
│   ├── run_glove_sentiment.py
│   ├── run_word2vec_sentiment.py
│   └── run_viz_best_combos.py
├── datasets/                    # evaluation datasets and semantic lexicons
│   ├── en-simlex.txt
│   ├── en-ws353.csv
│   ├── en_rg65.txt
│   ├── fr-simlex.dataset
│   ├── fr-ws353.dataset
│   ├── fr_rg65.txt
│   ├── ppdb/
│   │   └── ppdb-xl.txt
│   └── wolf/
│       └── wolf-1.0b4.xml
├── models/                      # original pretrained word embeddings
│   ├── glove.6B.300d.txt
│   ├── GoogleNews-vectors-negative300.bin.gz
│   └── cc.fr.300.vec
└── results/                     # experimental outputs, result tables, and figures
    ├── *_results.csv
    ├── *_bar.png
    └── *_curve.png
```

## Main Modules

### Core implementation

- `src/retrofit.py`: implements the retrofitting algorithm.
- `src/preprocessing.py`: builds semantic graphs from WordNet, PPDB, and WOLF; filters out-of-vocabulary nodes.
- `src/utils.py`: loads embeddings, saves embeddings, and computes cosine similarity.

### Resource preparation

- `resources_prepa/prepare_wordnet.py`: prepares English WordNet graphs.
- `resources_prepa/prepare_PPDB.py`: prepares PPDB lexical paraphrase graphs.
- `resources_prepa/prepare_french.py`: prepares French WOLF graphs.
- `resources_prepa/download_french_resources.py`: downloads French fastText and WOLF resources.
- `resources_prepa/download_word2vec.py`: downloads or prepares GoogleNews Word2Vec vectors.

### Evaluation

- `evaluation/eval_utils.py`: shared utilities for word similarity evaluation.
- `evaluation/run_glove_word_similarity.py`: evaluates GloVe vectors on English word similarity datasets.
- `evaluation/run_word2vec_word_similarity.py`: evaluates Word2Vec vectors on English word similarity datasets.
- `evaluation/run_french_word_similarity.py`: evaluates French fastText vectors on French word similarity datasets.
- `evaluation/run_glove_sentiment.py`: evaluates GloVe vectors on SST-2 sentiment classification.
- `evaluation/run_word2vec_sentiment.py`: evaluates Word2Vec vectors on SST-2 sentiment classification.
- `evaluation/run_viz_best_combos.py`: creates summary visualizations.

## Data

### Source links

The main resources used in this project are:

| Resource | Link |
|---|---|
| GloVe 6B embeddings | https://nlp.stanford.edu/projects/glove/ |
| GoogleNews Word2Vec | https://code.google.com/archive/p/word2vec/ |
| fastText French vectors | https://fasttext.cc/docs/en/crawl-vectors.html |
| WOLF French WordNet | https://almanach.inria.fr/software_and_resources/downloads/wolf-1.0b4.xml.bz2 |
| PPDB | https://github.com/mfaruqui/retrofitting/tree/master/lexicons |
| English WordNet | https://wordnet.princeton.edu/ |
| Stanford Sentiment Treebank / SST-2 | https://nlp.stanford.edu/sentiment/ |
| English WordSim353 | https://github.com/nmrksic/eval-multilingual-simlex/blob/master/evaluation/ws-353/wordsim353-english.txt |
| English SimLex999 | https://fh295.github.io/simlex.html |
| RG-65 multilingual datasets | https://www.site.uottawa.ca/~mjoub063/wordsims.htm |
| French multilingual word-pair datasets | https://github.com/siabar/Multilingual_Wordpairs |

### Pretrained embeddings

The project uses the following pretrained static embeddings:

| File | Description |
|---|---|
| `models/glove.6B.300d.txt` | English GloVe 6B, 300 dimensions |
| `models/GoogleNews-vectors-negative300.bin.gz` | English GoogleNews Word2Vec, 300 dimensions |
| `models/cc.fr.300.vec` | French fastText, 300 dimensions |

Large embedding files are stored in `models/`.

### Semantic resources

| File | Description |
|---|---|
| `datasets/ppdb/ppdb-xl.txt` | English PPDB lexical paraphrases |
| `datasets/wolf/wolf-1.0b4.xml` | WOLF, a French WordNet-like lexical resource |
| NLTK WordNet | English WordNet, loaded through `nltk.corpus.wordnet` |

### Evaluation datasets

| File | Language | Dataset |
|---|---|---|
| `datasets/en-ws353.csv` | English | WordSim-353 |
| `datasets/en-simlex.txt` | English | SimLex-999 |
| `datasets/en_rg65.txt` | English | RG-65 |
| `datasets/fr-ws353.dataset` | French | French WordSim-353 |
| `datasets/fr-simlex.dataset` | French | French SimLex |
| `datasets/fr_rg65.txt` | French | French RG-65 |

The French word-pair evaluation datasets are taken from the `siabar/Multilingual_Wordpairs` repository.

## OOV Handling

Semantic lexicons often contain words that are not present in the pretrained embedding vocabulary. Since retrofitting requires an initial vector for every updated word, these out-of-vocabulary words cannot be updated directly.

The project uses the following strategy:

1. Build the raw semantic graph from the lexical resource.
2. Keep only graph nodes that exist in the embedding vocabulary.
3. Remove edges connected to words without pretrained vectors.
4. Remove graph nodes with no remaining valid neighbors.

This creates a usable graph aligned with the embedding vocabulary.

## Usage

Run commands from the project root:

```bash
cd /path/to/retrofitting-wordvectors-final
```

### Prepare semantic graphs

English WordNet synonyms:

```bash
python3 resources_prepa/prepare_wordnet.py --relations syn
```

English WordNet synonyms, hypernyms, and hyponyms:

```bash
python3 resources_prepa/prepare_wordnet.py --relations all
```

English PPDB:

```bash
python3 resources_prepa/prepare_PPDB.py
```

French WOLF synonyms:

```bash
python3 resources_prepa/prepare_french.py --relations syn
```

French WOLF with hypernym links:

```bash
python3 resources_prepa/prepare_french.py --relations all
```

For quick tests, add:

```bash
--max-words 50000
```

### Search alpha

English alpha search on SimLex:

```bash
python3 src/run_english_alpha_search.py
```

French alpha search on French SimLex:

```bash
python3 src/run_french_alpha_search.py
```

### Run word similarity evaluation

English GloVe:

```bash
python3 evaluation/run_glove_word_similarity.py
```

English Word2Vec:

```bash
python3 evaluation/run_word2vec_word_similarity.py
```

French fastText:

```bash
python3 evaluation/run_french_word_similarity.py
```

### Run sentiment evaluation

GloVe on SST-2:

```bash
python3 evaluation/run_glove_sentiment.py
```

Word2Vec on SST-2:

```bash
python3 evaluation/run_word2vec_sentiment.py
```

## Current Experimental Settings

The main experiments use:

- `num_iters = 10`;
- `beta_strategy = "inverse_degree"`;
- English best alpha: `0.429`, selected on English SimLex;
- French best alpha: `2.333`, selected on French SimLex;
- English resources: WordNet synonyms, WordNet all relations, PPDB;
- French resource: WOLF.

## Results

### Alpha search

**English — GloVe + WordNet synonyms, SimLex-999**

| alpha | weight on original | retrofitted ρ | Δ vs baseline |
|---:|---:|---:|---:|
| 0.111 | 0.10 | 0.3879 | +0.0174 |
| 0.250 | 0.20 | 0.4456 | +0.0751 |
| **0.429** | **0.30** | **0.4576** | **+0.0871** |
| 0.667 | 0.40 | 0.4543 | +0.0838 |
| 1.000 | 0.50 | 0.4454 | +0.0749 |
| 1.500 | 0.60 | 0.4335 | +0.0630 |
| 2.333 | 0.70 | 0.4194 | +0.0489 |
| 4.000 | 0.80 | 0.4042 | +0.0337 |
| 9.000 | 0.90 | 0.3877 | +0.0172 |

Best English α = **0.429** 

**French — fastText + WOLF synonyms, French SimLex**

| alpha | weight on original | retrofitted ρ | Δ vs baseline |
|---:|---:|---:|---:|
| 0.111 | 0.10 | 0.2736 | −0.0470 |
| 0.250 | 0.20 | 0.3085 | −0.0121 |
| 0.429 | 0.30 | 0.3318 | +0.0112 |
| 0.667 | 0.40 | 0.3445 | +0.0239 |
| 1.000 | 0.50 | 0.3525 | +0.0319 |
| 1.500 | 0.60 | 0.3550 | +0.0344 |
| **2.333** | **0.70** | **0.3552** | **+0.0346** |
| 4.000 | 0.80 | 0.3485 | +0.0279 |
| 9.000 | 0.90 | 0.3373 | +0.0166 |

Best French α = **2.333** 

### Word similarity

English GloVe 6B 300d:

| Combination | Dataset | Original rho | Retrofitted rho | Delta |
|---|---|---:|---:|---:|
| GloVe + WN_syn | WS-353 | 0.6085 | 0.6074 | -0.0012 |
| GloVe + WN_syn | RG-65 | 0.7695 | 0.7773 | +0.0078 |
| GloVe + WN_all | WS-353 | 0.6085 | 0.6040 | -0.0045 |
| GloVe + WN_all | RG-65 | 0.7695 | 0.8744 | **+0.1048** |
| GloVe + PPDB | WS-353 | 0.6085 | 0.5326 | -0.0759 |
| GloVe + PPDB | RG-65 | 0.7695 | 0.8074 | +0.0379 |

English Word2Vec GoogleNews 300d:

| Combination | Dataset | Original rho | Retrofitted rho | Delta |
|---|---|---:|---:|---:|
| Word2Vec + WN_syn | WS-353 | 0.6962 | 0.6320 | -0.0642 |
| Word2Vec + WN_syn | RG-65 | 0.7608 | 0.7682 | +0.0074 |
| Word2Vec + WN_all | WS-353 | 0.6962 | 0.6681 | -0.0280 |
| Word2Vec + WN_all | RG-65 | 0.7608 | 0.8816 | **+0.1208** |

French fastText cc.fr.300 + WOLF:

| Combination | Dataset | Original rho | Retrofitted rho | Delta |
|---|---|---:|---:|---:|
| fastText + WOLF_syn | WS-353_FR | 0.5328 | 0.5033 | -0.0295 |
| fastText + WOLF_syn | RG-65_FR | 0.8258 | 0.8377 | +0.0119 |
| fastText + WOLF_all | WS-353_FR | 0.5328 | 0.4988 | -0.0340 |
| fastText + WOLF_all | RG-65_FR | 0.8258 | 0.8500 | +0.0242 |

### Sentiment Analysis in English

SST-2 is evaluated with sentence vectors obtained by averaging word vectors and training a logistic regression classifier.

| Combination | Original accuracy | Retrofitted accuracy | Delta |
|---|---:|---:|---:|
| GloVe + WN_syn | 0.7798 | 0.7787 | -0.0011 |
| GloVe + WN_all | 0.7798 | 0.7626 | -0.0172 |
| GloVe + PPDB | 0.7798 | 0.7649 | -0.0149 |
| Word2Vec + WN_syn | 0.7982 | 0.7982 | 0.0000 |
| Word2Vec + WN_all | 0.7982 | 0.7993 | +0.0011 |

## Interpretation

The results show that retrofitting is most useful for datasets that measure semantic similarity, especially RG-65 and SimLex-style evaluations. Performance on WS-353 can decrease because WS-353 includes broader topical relatedness, not only strict functional similarity. Sentiment classification also does not always improve, since averaging static word vectors is a simple sentence representation and the semantic graph is not directly optimized for sentiment.

Overall, the implementation reproduces the main behavior described by Faruqui et al. (2015): semantic lexicons can improve static word vectors when the evaluation task benefits from synonymy, hypernymy, or related lexical information.

## Reference

Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., and Smith, N. A. (2015). *Retrofitting Word Vectors to Semantic Lexicons*.

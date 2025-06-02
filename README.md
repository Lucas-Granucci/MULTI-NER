# MULTI-NER: Multi-Domain Named Entity Recognition

## Overview
This is the official code repository for the research project "Breaking Language Barriers: Cross-lingual Transfer Learning and Pseudo-Labeling for Natural Language Processing in Low-Resource Languages" found here -> Paper

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Architectures](#model-architecture)
- [Results](#results)

## Introduction
Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying entities in text. This project improves performance of NER systems for low-resource languages by transfering knowledge from adjacent languages and by iterativley training on synthethically generated pseudo-labels.

## Installation
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/Lucas-Granucci/MULTI-NER.git
cd MULTI-NER
conda env create -f environment.yml
```

## Usage
Sections in 
```_MAIN_.ipynb```
include the training of the baseline BERT-BiLSTM-CRF model, testing various ratios of high to low-resource language data in cross-lingual transfer learning, and testing various confidence intervals in iterative pseudo-labeling.

## Dataset
The data for this project is sourced from the [WikiANN](https://huggingface.co/datasets/unimelb-nlp/wikiann) multi-lingual NER dataset. The specific language pairs used and their are presented below:
| Low-Resource Language | High-Resource Language | Low-Resource Sentence Count | High-Resource Sentence Count |
|-----------------------|------------------------|----------------------------|------------------------------|
| Malagasy              | Indonesian             | 300                        | 40,000                       |
| Faroese               | Danish                 | 300                        | 40,000                       |
| Corsican              | Italian                | 300                        | 40,000                       |
| Upper Sorbian         | Polish                 | 300                        | 40,000                       |
| Bhojpuri              | Hindi                  | 300                        | 7,000                        |
| Chuvash               | Turkish                | 300                        | 40,000                       |
 
## Model Architecture
The backbone model architecture used is BERT-BiLSTM-CRF. The BERT layer is used to generate representative embeddings for input tokens and BiLSTM-CRF was shown to be effective for sequence taggings tasks [(Huang et al., 2015)](https://arxiv.org/abs/1508.01991)

## Results
Overall, this research finds cross-lingual transfer learning and the proposed self-training technique to be highly effective at improving the performance of NER models for low-resource languages.

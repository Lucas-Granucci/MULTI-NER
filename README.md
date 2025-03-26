# MULTI-NER: Multi-Domain Named Entity Recognition

## Overview
This project focuses on developing a multi-domain Named Entity Recognition (NER) system. The goal is to create a model that can accurately identify and classify named entities across various domains.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Architectures](#model-architecture)
- [Results](#results)
- [Contact](#contact)

## Introduction
Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying entities in text. This project aims to enhance NER performance across multiple domains by leveraging advanced machine learning techniques.

## Installation
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/Lucas-Granucci/MULTI-NER.git
cd MULTI-NER
conda env create -f environment.yml
```

## Usage
```_MAIN_.ipynb```

## Datasets
The data for this project is sourced from the [WikiANN](https://huggingface.co/datasets/unimelb-nlp/wikiann) multi-lingual NER dataset.

## Model Architecture
The model architectures being tested is Bert-Bilstm-Crf.

## Results
The results of the experiments are documented in the ```_RESULTS_.ipynb```. Key performance metrics include train and validation F1-scores.

## Contact
For any questions or inquiries, please contact Lucas Granucci at lucasgranucci08@gmail.com.
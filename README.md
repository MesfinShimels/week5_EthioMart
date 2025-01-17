# week5_EthioMart
# EthioMart NER Project

## Overview
This project is aimed at creating a centralized platform for e-commerce activities in Ethiopia. It includes tools for data ingestion, preprocessing, and training a Named Entity Recognition (NER) model for Amharic text.

## Folder Structure
EthioMart_NER_Project/
│
├── data/                # Raw and preprocessed datasets
│   ├── raw/             # Raw scraped data
│   ├── preprocessed/    # Cleaned and tokenized data
│   └── labeled/         # Labeled data in CoNLL format
│
├── notebooks/           # Jupyter notebooks for prototyping
│   └── data_ingestion.ipynb
│
├── scripts/             # Python scripts for reusable code
│   ├── scraper.py       # Telegram scraper for data ingestion
│   ├── preprocessing.py # Preprocessing pipeline
│   ├── labeling.py      # Script for dataset labeling
│   └── train_model.py   # Model fine-tuning and evaluation
│
├── models/              # Saved models and checkpoints
│   ├── base_model/      # Pretrained base model
│   └── fine_tuned/      # Fine-tuned model for Amharic NER
│
├── outputs/             # Outputs like predictions and logs
│   ├── predictions/     # Model outputs
│   └── logs/            # Log files
│
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and instructions
└── .env                 # Environment variables for Telegram API keys

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

EthioMart NER Project
Overview
EthioMart is an innovative project aimed at creating a centralized platform for e-commerce activities in Ethiopia. The project focuses on building tools for data ingestion, preprocessing, and training a Named Entity Recognition (NER) model to process Amharic text. By leveraging this model, EthioMart aims to extract meaningful information from text data, such as product details, prices, and seller information, to enhance the online shopping experience in Ethiopia.
Folder Structure
EthioMart_NER_Project/
│
├── .vscode/             # VSCode-specific configurations (e.g., launch.json, settings.json)
│
├── Cleaned_Data/        # Repository for cleaned datasets
│
├── data/                # Raw and preprocessed datasets
│   ├── raw/             # Raw scraped data
│   │   ├── labeled_telegram_product_price_logs.xlsx
│   │   └── telegram_data.xlsx
│   ├── preprocessed/    # Cleaned and tokenized data
│   └── labeled/         # Labeled data in CoNLL format
│
├── models/              # Saved models and checkpoints
│   ├── base_model/      # Pretrained base model
│   └── fine_tuned/      # Fine-tuned model for Amharic NER
│
├── notebooks/           # Jupyter notebooks for prototyping and experimentation
│   ├── data_ingestion.ipynb      # Notebook for scraping and ingesting raw data
│   └── main_pipeline.ipynb       # End-to-end pipeline for data processing and model training
│
├── outputs/             # Outputs like predictions and logs
│   ├── logs/            # Log files for tracking experiments and processes
│   └── predictions/     # Model outputs and predictions
│
├── screen shots/        # Screenshots for visual documentation and demos
│
├── scripts/             # Python scripts for reusable code
│   ├── labeling.py                 # Script for dataset labeling
│   ├── model_comparison.py         # Script for comparing different NER models
│   ├── model_interpretability.py   # Tools for model explainability
│   ├── preprocessing.py            # Preprocessing pipeline for raw data
│   ├── scraper.py                  # Data scraper for extracting Telegram data
│   ├── telegram_scrapper.py        # Alternative script for Telegram scraping
│   └── train_model.py              # Script for training and evaluating the NER model
│
├── src/                 # Source code for modular components
│
├── tests/               # Unit and integration tests
│
├── .env                 # Environment variables for Telegram API keys and other configurations
├── .gitignore           # List of files and directories to ignore in version control
├── README.md            # Project overview and instructions
└── requirements.txt     # Python dependencies
Features
•	Data Scraping: Scrapes product and price data from Telegram channels.
•	Data Preprocessing: Tokenizes and cleans Amharic text data for NER tasks.
•	Data Labeling: Provides tools for annotating datasets in CoNLL format.
•	NER Model Training: Fine-tunes a pretrained language model for Amharic NER.
•	Model Evaluation: Includes scripts for comparing model performance and interpretability.
Installation
1.	Clone the repository:
2.	git clone https://github.com/mesfinshimels/EthioMart_NER_Project.git
3.	cd EthioMart_NER_Project
4.	Create a virtual environment:
5.	python -m venv env
6.	source env/bin/activate  # On Windows: env\Scripts\activate
7.	Install dependencies:
8.	pip install -r requirements.txt
9.	Set up environment variables in the .env file:
10.	TELEGRAM_API_ID=your_api_id
11.	TELEGRAM_API_HASH=your_api_hash
Dependencies
Ensure the following libraries are included in requirements.txt:
•	numpy
•	pandas
•	matplotlib
•	seaborn
•	scikit-learn
•	torch
•	transformers
•	spacy
•	flair
•	tqdm
•	telethon
•	dotenv
•	jupyter
•	pytest
Usage
Data Scraping
Run the Telegram scraper to collect raw data:
python scripts/scraper.py
Preprocessing
Clean and tokenize the raw data:
python scripts/preprocessing.py
Dataset Labeling
Label the preprocessed data using the labeling script:
python scripts/labeling.py
Model Training
Fine-tune the NER model using:
python scripts/train_model.py
Model Evaluation
Compare model performance:
python scripts/model_comparison.py
Screenshots
Screenshots of the project pipeline and outputs are available in the screen shots/ directory.
Contributing
Contributions are welcome! Please follow these steps:
1.	Fork the repository.
2.	Create a new branch (feature/new-feature).
3.	Commit your changes.
4.	Push the branch.
5.	Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or suggestions, feel free to reach out:
•	Email: mesfins@gmail.com
•	

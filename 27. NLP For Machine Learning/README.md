# NLP For Machine Learning

This repository contains a practical progression through Natural Language Processing (NLP) techniques used in machine learning. Each numbered folder focuses on one concept and includes a Jupyter notebook with examples, explanations, and exercises.

## What This Repository Covers

- Text tokenization and preprocessing
- Stemming, lemmatization, and stopword removal
- Part-of-speech tagging and named entity recognition
- One-hot encoding and bag-of-words representations
- N-grams and TF-IDF feature extraction
- Word embeddings and Word2Vec intuition
- Text classification using BOW, TF-IDF, Word2Vec, and averaged Word2Vec vectors
- Spam and ham classification
- Kindle review sentiment analysis

## Repository Structure

```text
27. NLP For Machine Learning/
|
|-- 1.Tokenization/
|-- 2.Text Preprocessing Stemming using NLTK/
|-- 3.Lemitization/
|-- 4.Text Preprocessing-Stopwords with NLTK/
|-- 5.Part of Speech Tagging/
|-- 6.Named Entity Recognition/
|-- 7. What's Next/
|-- 8.One Hot Encoding/
|-- 9.Bag of words/
|-- 10.N grams/
|-- 11. TF-IDF Instituion/
|-- 12.Word Embeddings/
|-- 13.Word2vec Intution/
|-- 14.Spam ham Project using BOW/
|-- 15.Spam And Ham Project Using TFidf/
|-- 16. Part I-Text Classification With Word2vec And AvgWord2vec/
|-- 17.part-1Kindle REview Sentiment Analysis/
`-- pdfs/
```

Most topic folders contain a `main.ipynb` notebook. The project folders also include the datasets used by their notebooks:

- `SMSSpamCollection.csv` for spam and ham classification examples
- `all_kindle_review.csv` for Kindle review sentiment analysis

The `pdfs` folder contains supporting reference material.

## Recommended Learning Order

Follow the folders in their numeric order. The early notebooks introduce NLP fundamentals, the middle notebooks explain common text-vectorization methods, and the final notebooks apply those methods to complete machine-learning projects.

## Getting Started

1. Install Python 3.9 or later.
2. Install Jupyter Notebook or use Visual Studio Code with the Jupyter extension.
3. Install the packages required by the notebooks, commonly including:

   ```bash
   pip install jupyter nltk numpy pandas scikit-learn matplotlib seaborn gensim
   ```

4. Open a notebook, run the setup cells first, and execute the remaining cells from top to bottom.
5. If an NLTK resource is missing, download it from a Python cell as directed by the notebook, for example:

   ```python
   import nltk
   nltk.download("punkt")
   ```

## Notes

- Run notebooks from their own folders when loading local CSV files.
- Notebook outputs may depend on the installed Python and library versions.
- Folder names are kept as they appear in the original course material.

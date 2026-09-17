📄 [Full Project Report](NLP_Project_Report_Aditya.pdf) | 📓 [Notebook](NLP_BERT_Sentiment_Notebook.ipynb)

# Multi-Domain Sentiment Analysis using BERT with Transfer Learning Evaluation

Compares classical ML models (Naive Bayes, Logistic Regression, SVM) against a fine-tuned BERT model for sentiment classification across three domains: movie reviews (IMDB), product reviews (Amazon), and tweets (TweetEval).

## Results
| Model | Accuracy | F1-Score |
|---|---|---|
| Naive Bayes | 75.02% | 75.46% |
| Logistic Regression | 79.03% | 79.14% |
| SVM (Linear) | 78.59% | 78.55% |
| **BERT (Transfer Learning)** | **86.22%** | **86.26%** |

**Live Demo:** [Coming in Step 2]

## Key Finding
BERT outperforms all classical models by ~7 percentage points, but shows a notable accuracy drop (71% vs 93%) on short, informal text (tweets) — driven primarily by confusion around the Neutral sentiment class.

## Tech Stack
Python, Hugging Face Transformers, PyTorch, scikit-learn, Gradio

## Author
Aditya Gupta — BSc Data Science, KES Shroff College

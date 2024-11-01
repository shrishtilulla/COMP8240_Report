# COMP8240_Report
Final group assessemnt - code for the data analysis for ULiMFiT
Sentiment Analysis of Kindle Reviews
This project implements a sentiment analysis model for Kindle reviews using the Fastai library. The model applies the Universal Language Model Fine-tuning for Text Classification (ULMFiT) approach, leveraging the AWD-LSTM architecture to classify reviews into three sentiment categories: negative, neutral, and positive.

Table of Contents
Introduction
Installation
Dataset
Usage
Model Training
Results
Acknowledgments
Introduction
This project analyzes customer sentiments from Kindle reviews using ULMFiT, a transfer learning approach that fine-tunes a pre-trained language model for specific tasks like sentiment classification. This technique improves model adaptability, making it well-suited for accurate sentiment analysis on Kindle product reviews.

Installation
To run this project, install the following dependencies:

bash
Copy code
pip install fastai
pip install pandas
pip install scikit-learn
Dataset
The dataset includes Kindle review text and corresponding ratings. Ratings are categorized as follows:

Negative: Ratings 1 and 2
Neutral: Rating 3
Positive: Ratings 4 and 5
Usage
Load and preprocess the dataset:

python
Copy code
import pandas as pd

df = pd.read_csv('path/to/kindle_reviews.csv')
Preprocess the data and create DataLoaders for training and validation:

python
Copy code
from fastai.text.all import *

dls = TextDataLoaders.from_df(
    df,
    text_col='reviewText',
    label_col='sentiment',
    valid_pct=0.2,
    bs=16
)
Model Training
Using ULMFiT for Sentiment Analysis
ULMFiT’s multi-stage training process is employed for optimal performance. This includes:

Language Model Fine-Tuning: The AWD-LSTM model, fine-tuned on the Kindle dataset.
Text Classification: Applying ULMFiT’s fine-tuning steps with freezing and unfreezing layers for incremental learning.
python
Copy code
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fit_one_cycle(10)

# Fine-tune by freezing and unfreezing layers
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2))
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))
Results
Model performance is evaluated with precision, recall, and F1-score. Sample classification results:

Metric	Negative	Neutral	Positive
Precision	0.90	0.00	0.74
Recall	0.78	0.00	0.86
F1-score	0.84	0.00	0.79
Overall accuracy: 74%


#Conclusion
We are able to see the model improve from 200 reviews to 1000 reviews because with more learning and diverse prompts the model is understanding the sentiments but struggles to rule our neutral emotions that well.
Like it ccan distinguish if somethin gis in between good or bad in classification.

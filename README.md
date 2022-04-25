# Twitter-Sentiment-Analysis
SC1015 Introduction to Data Science and Artificial Intelligence Mini-Project

## About

Mini-project for SC1015 - "Data Science and Artificial Intelligence"  focusing on the detection of hate speech in tweets using DS and NLP concepts. 

The dataset can be found here:
https://www.kaggle.com/datasets/dv1453/twitter-sentiment-analysis-analytics-vidya?select=train_E6oV3lV.csv

## Contributors
* Bhat Sachin   &#8594; @Sachin-Bhat
* Nalin Sharma  &#8594; @nalin0503

## Motivations/Problem Statement

Motivations: In a democratic context, the right to free speech is deemed essential by many. People wish to voice their opinions on key decisions, capturing the essence of a democracy. This fundamental right can be used for promoting collaborative action, spreading awareness and fostering a two-way communication between the citizens of a country and its government. However, we must consider the flip side - the inclusion of derogatory, hurtful and biased opinion on a public platform may be the bad apple that can plague the collective mindset of our societies, negatively affecting them in ways that may be irreversible. The presence of hate speech online can materialise itself into physical hate crimes, and so it is probable that the government may wish to regulate the online presence of its citizens. If this were to happen, what would be the best approach algorithmically? 

Problem statement/ definition - Effective implementation of Data Science and Natural Language Processing (NLP) concepts to find the best model to detect hate speech in tweets. 

How can we effectively detect hate speech in tweets?

## Features
- Bag-Of-Words
- TF-IDF (Term Frequency - Inverse Document Frequency)
- Word Embeddings
- Word2Vec
- Doc2Vec

## Models (Classification)
- Support Vector Machine (SVM)
- Logistic Regression (LReg)
- RandomForest (RF)
- XGBoost (XGB)

## Conclusion
- Overall, XGBoost turned out to be the best module
- Because it works by boosting the tree towards the best solution i.e. it is a greedy algorithm
- Specifically, Word2Vec was the best parameter due to the volume of data points available
- We further tried to optimise the XGBoost model using hyperparameter tuning and grid search
- This gave us better f1 scores.
- Furthermore these predictions when processed could be useful for analysing hate crime motives.

## Limitations
1. The program may take a long time to run due to the high number of epochs and the large sample size. You may reduce either one or both if you specifically need faster results, although that would compromise accuracy. 
2. For hyperparameter tuning, the update sequence is manual. 
  


## Reflections/Learning Points
1. Acquired knowledge on the interconnectedness between jupyter notebook, VSCode and GitHub.
2. Learnt about the functionalities of the programs stated above. 
3. Soft skills - learnt how to present a DSAI project in a structured, articulate manner, training us for our professional capacities in the future. 
4. Performing Data Prep, Cleaning and EDA on a large textual dataset.
5. Basics of 'text mining' in general.   
6. An understanding of APIs and its documentation.
7. Natural Language Processing concepts such as text normalisation, wordclouds to represent data, extracting features from tokenised strings, word embeddings and the workings of the various models as stated in previous sections. 
8. Computation of F1 Scores 
9. Use of added modules such as gensim and PorterStemmer to aid our project

## Contributions
* Bhat Sachin   &#8594; Data Collection, Model Building for SVM and XGBoost, Feature Extraction, Hyperparameter Tuning
* Nalin Sharma  &#8594; Data Preparation, Cleaning, EDA, Model Building Logistic Regression and RandomForest, Presentation slides and script

## References 
1. https://www.washingtonpost.com/nation/2018/11/30/ 
2. how-online-hate-speech-is-fueling-real-life-violence/
3. https://time.com/6121915/reddit-international-hate-speech/
4. https://scikit-learn.org/stable/
5. https://docs.python.org/3/
6. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
7. https://monkeylearn.com/blog/what-is-tf-idf/
8. https://medium.com/red-buffer/doc2vec-computing-similarity-between-the-documents-47daf6c828cd
9. https://www.educative.io/edpresso/what-is-the-f1-score
10. https://machinelearningmastery.com/gentle-introduction-bag-words-model/
11. https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
12. https://anchormen.nl/blog/digital-transformation/accuracy-precision-recall-models/
13. https://hackinghate.eu/news/when-online-hate-speech-goes-extreme-the-case-of-hate-crimes/
14. https://www.kdnuggets.com/2020/12/xgboost-what-when.html
https://cloud.google.com/ai-platform/training/docs/hyperparameter-tuning-overview



## General Setup Instructions
Not all modules are available by default in the Anaconda Navigator package environment. For the project to be run on your system, kindly add `conda-forge` to your list of channels as shown below.

<img width="959" alt="2022-04-23_11-31" src="https://user-images.githubusercontent.com/25080916/164883067-c2373b53-b771-43a2-8d9d-da5678368c5c.png">

When a module needs to be installed, please install it by running the following command in a terminal: 

<code>conda install *name-of-module*</code>

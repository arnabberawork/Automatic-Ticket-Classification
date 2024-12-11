# Automatic Ticket Classification
> Machine learning model using NLP topic modeling to automatically classify customer complaints based on products and services for improved customer service in the financial industry.


## Table of Contents :
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Approach](#approach)
* [Technologies/Libraries Used](#technologies/libraries-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Glossary](#glossary)
* [Author](#author)


## Problem Statement
For a financial company, customer complaints carry a lot of importance, as they are often an indicator of the shortcomings in their products and services. If these complaints are resolved efficiently in time, they can bring down customer dissatisfaction to a minimum and retain them with stronger loyalty. This also gives them an idea of how to continuously improve their services to attract more customers. 
These customer complaints are unstructured text data; so, traditionally, companies need to allocate the task of evaluating and assigning each ticket to the relevant department to multiple support employees. This becomes tedious as the company grows and has a large customer base.
In this case study, you will be working as an NLP engineer for a financial company that wants to automate its customer support tickets system. As a financial company, the firm has many products and services such as credit cards, banking and mortgages/loans. 

## Objectives
You need to build a model that is able to classify customer complaints based on the products/services. By doing so, you can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.
With the help of non-negative matrix factorization (NMF), an approach under topic modelling, you will detect patterns and recurring words present in each ticket. This can be then used to understand the important features for each cluster of categories. By segregating the clusters, you will be able to identify the topics of the customer complaints. 
You will be doing topic modelling on the .json data provided by the company. Since this data is not labelled, you need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:
    - Credit card / Prepaid card
    - Bank account services
    - Theft/Dispute reporting
    - Mortgages/loans
    - Others 

With the help of topic modelling, you will be able to map each ticket onto its respective department/category. You can then use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, you can classify any new customer complaint support ticket into its relevant department.
Download the dataset given below.

The data set given to you is in the .json format and contains 78,313 customer complaints with 22 features. You need to convert this to a dataframe in order to process the given complaints.

[Dataset]( https://github.com/arnabberawork/Automatic-Ticket-Classification/blob/main/complaints-2021-05-14_08_16.json )

## Approach

- Step 1: Workspace set up: Import and Install Necessary Libraries
- Step 2: Load the Data and Understanding the Data
- Step 3: Text preprocessing
- Step 4: Exploratory data analysis (EDA)
- Step 5: Feature extraction
- Step 6: Topic modelling 
- Step 7: Model building using supervised learning
- Step 8: Model training and evaluation
- Step 9: Model inference
- Step 10: Evaluation
- Step 11: Making Predictions Using the Model

<b>Note</b>: Once you have finalised the clusters/categories for customer complaints, the next step is to create a data set that contains the complaints and labels (which you found using NMF). This labelled data set will be used for model building using supervised learning. 
You need to try at least any three models from logistic regression, naive Bayes, decision tree and random forest. 
You need to select the model that performs the best according to the evaluation metrics.
  
## Technologies/Libraries Used
    - Python: 3.12.3
    - Numpy: 1.26.4
    - Pandas: 2.2.2
    - NLTK: 3.9.1
    - SpaCy: 3.8.2
    - Seaborn: 0.13.2
    - Matplotlib: 3.9.2
    - Plotly: 5.24.1
    - Scikit-learn: 1.5.1

## Conclusions
In this project, two models were developed to extract diseases and treatments from unstructured medical text:

Model 1 (Using POS as a feature): Achieved an F1 score of 0.907.

Model 2 (Using both POS and Dependency Parsing as features): Achieved an F1 score of 0.904.

Both models performed well, with Model 1 slightly outperforming Model 2. The use of POS tagging and dependency parsing as features contributed positively to the extraction process, demonstrating the effectiveness of these linguistic features in improving NER performance.

Further Steps/Improvements:

To enhance the model further, consider exploring the following:

Additional Features: Experiment with adding other features such as word embeddings (e.g., Word2Vec, GloVe) or character-level features (e.g., character n-grams) to capture more semantic information.

Deep Learning Models: Try using more advanced deep learning-based NER models like BERT or BioBERT, which are specifically designed for medical text and might improve accuracy.

Entity Linking: Add a post-processing step for linking the extracted entities to a medical knowledge base (e.g., UMLS) for better contextual understanding.

## Acknowledgements

- The project reference course materieals from upGrads curriculm .
- The project references from presentation in upGrads module given by [Alankar Gupta](https://www.linkedin.com/in/alankar-gupta-898a9659/)
- The project references insights and inferences from presentation in upGrads live class given by [Dr. Apurva Kulkarni] (https://www.linkedin.com/in/dr-apurva-kulkarni-33a074189/)
- The project references from presentation in upGrads live class given by [Amit Pandey](https://www.linkedin.com/in/amitpandeyprofile/)

## Glossary

- Data Preparation
    - Column Renaming
    - Null Value Removal
    - Lowercasing
    - Text Cleaning (e.g., punctuation and square bracket removal)
    - Removing Words with Numbers
- Text Processing
    - Lemmatization
    - Part-of-Speech (POS) Tagging
- Exploratory Data Analysis (EDA)
    - Character Length Visualization
    - Word Cloud Generation
    - N-gram Frequency Analysis
- Feature Engineering
    - Feature Extraction
    - Topic Modeling
    - Non-Negative Matrix Factorization - NMF
- Model Building
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - Naive Bayes Classifier
    - XGBoost Classifier
- Additional Concepts & Tools
    - Tokenization
    - spaCy
    - Train-Test Split
    - Model Evaluation (e.g., F1 Score)
    - Pickle (for model saving)


## Author
* [Arnab Bera]( https://www.linkedin.com/in/arnabbera-tech/ )

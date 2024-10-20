# Text Classification Model Comparison with Preprocessing

## Project Overview

This project focuses on comparing the performance of different classification models on text data using multiple preprocessing techniques. It aims to evaluate several text vectorization strategies (CountVectorizer and TfidfVectorizer) combined with different classifiers (Logistic Regression, Decision Tree, Multinomial Naive Bayes) to assess their effectiveness in terms of accuracy and speed. Additionally, it incorporates text preprocessing techniques such as stemming and lemmatizing to normalize the text data before classification.

### Dataset

The dataset used in this project contains two columns:
- `text`: The text to be classified.
- `humor`: A boolean label indicating whether the text is humorous (`True`) or not (`False`).

### Goal

The primary goal is to identify the best-performing model based on accuracy and the best parameter settings using GridSearchCV. The models will be evaluated using 5-fold cross-validation, and the results of each model will be compared based on their accuracy and time to fit the data.

---

## Project Structure

- **`TextCM.ipynb`**: The main Jupyter notebook where the entire experiment is conducted.
- **`dataset.csv`**: The full dataset containing the text and humor labels used for training and testing the models.
- **`dataset-minimal.csv`**: A minimal version of the dataset with fewer records (useful for testing purposes).
- **`README.md`**: This file, which explains the project setup, usage, and instructions.

---

## Project Overview

This project focuses on comparing the performance of different classification models on text data using multiple preprocessing techniques. It aims to evaluate several text vectorization strategies (CountVectorizer and TfidfVectorizer) combined with different classifiers (Logistic Regression, Decision Tree, Multinomial Naive Bayes) to assess their effectiveness in terms of accuracy and speed. Additionally, it incorporates text preprocessing techniques such as stemming and lemmatizing to normalize the text data before classification.

### Dataset

The dataset used in this project contains two columns:

- `text`: The text to be classified.

- `humor`: A boolean label indicating whether the text is humorous (True) or not (False).

### Goal

The primary goal is to identify the best-performing model based on accuracy and the best parameter settings using GridSearchCV. The models will be evaluated using 5-fold cross-validation, and the results of each model will be compared based on their accuracy and time to fit the data.

### Project Structure

- `TextCM.ipynb`: The main Jupyter notebook where the entire experiment is conducted.

- `dataset.csv`: The full dataset containing the text and humor labels used for training and testing the models.

- `dataset-minimal.csv`: A minimal version of the dataset with fewer records (useful for testing purposes).

- `README.md`: This file, which explains the project setup, usage, and instructions.

### Requirements

To run this project, you'll need the following libraries:

- Python 3.7+

- pandas

- scikit-learn

- nltk

- matplotlib

Additionally, you may need to download NLTK resources (for lemmatization):


### Key Steps in the Notebook

#### Text Preprocessing:

Perform both stemming using NLTK’s SnowballStemmer and lemmatizing using WordNetLemmatizer to normalize the text.

#### Vectorization:

Use CountVectorizer and TfidfVectorizer to transform text data into numerical features. Parameters like stop_words and max_features are optimized.

#### Classification:

Use Logistic Regression, Decision Tree, Multinomial Naive Bayes, and Bernoulli Naive Bayes classifiers.

Compare the classifiers using cross-validation and GridSearchCV to evaluate their performance in terms of accuracy and speed.

#### Evaluation:

##### Compare the models based on accuracy and fitting time.

Visualize the results with bar charts to understand the trade-off between accuracy and fitting time for each combination of preprocessing method, vectorizer, and classifier.

##### Results and Conclusion

After running the notebook, you will get a summary table of the best models, their parameters, and the accuracy scores. You can use this to determine the most suitable classifier for your text classification task.

##### Project Enhancements

Additional Preprocessing: In the future, you can add more advanced text preprocessing techniques, such as removing punctuation, or using more advanced vectorization strategies like word embeddings.

More Classifiers: You can extend the comparison by adding more classifiers, such as RandomForestClassifier, SVM, or even deep learning models.

#### Acknowledgments

NLTK for providing robust text preprocessing tools.

scikit-learn for making machine learning easy with its rich set of APIs for classification and model tuning.

matplotlib for providing tools to create insightful visualizations.


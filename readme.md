# SMS Spam Detection using Machine Learning

This project is a Jupyter Notebook for detecting spam SMS messages using Natural Language Processing (NLP) and machine learning techniques.

## Features

- **Data Loading & Cleaning:**  
  Loads the SMS Spam Collection dataset, removes unnecessary columns, handles missing and duplicate values, and cleans the text data (removes HTML, punctuation, stopwords, and applies stemming).

- **Exploratory Data Analysis (EDA):**  
  - Visualizes the distribution of spam and ham messages.
  - Analyzes the most common words in the dataset using bar charts.

- **Natural Language Processing (NLP):**  
  - Tokenizes and preprocesses text data.
  - Removes stopwords and applies stemming for normalization.

- **Feature Extraction:**  
  - Uses Bag of Words (CountVectorizer) to convert text into numerical features for machine learning.

- **Model Training:**  
  - Splits the data into training and testing sets.
  - Trains a Logistic Regression classifier to distinguish between spam and ham messages.

- **Evaluation & Visualization:**  
  - Predicts on the test set and calculates accuracy.
  - Displays a confusion matrix as a heatmap.
  - Prints a detailed classification report (precision, recall, F1-score).

## Results

- **Accuracy:**  
  The model achieves high accuracy (typically above 95%) in classifying SMS messages as spam or ham.

- **Confusion Matrix:**  
  Shows the number of true positives, true negatives, false positives, and false negatives, helping you understand the model’s performance on each class.

- **Classification Report:**  
  Provides precision, recall, and F1-score for both spam and ham classes, indicating strong performance in both detecting spam and avoiding false positives.

- **Visual Insights:**  
  - Bar plots for class distribution and most common words help understand the dataset.
  - Heatmap of the confusion matrix visually summarizes prediction results.

## Usage

1. **Download the dataset:**  
   Place `spam.csv` in the same directory as the notebook.

2. **Run the notebook:**  
   Open `ML (1).ipynb` in Jupyter Notebook or VS Code and run all cells.

3. **Colab:**  
   [![Open In Colab](https://colab.research.google.com/github/msaswata15/emailsp/blob/main/ML.ipynb)](https://colab.research.google.com/github/msaswata15/emailsp/blob/main/ML.ipynb)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- tqdm

Install dependencies with:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn nltk tqdm
```

## Project Structure

- `ML (1).ipynb` — Main notebook with code and analysis
- `spam.csv` — SMS Spam Collection dataset

## License

For educational purposes

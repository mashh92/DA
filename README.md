1----
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create 'Position_Salaries' dataset
data = {'Position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]}

df = pd.DataFrame(data)

# Identify independent (X) and target (y) variables
X = df[['Level']]
y = df['Salary']

# Split the variables into training and testing sets (7:3 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the training and testing sets
print("Training set:")
print("X_train:\n", X_train)
print("y_train:\n", y_train)

print("\nTesting set:")
print("X_test:\n", X_test)
print("y_test:\n", y_test)

# Build a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error on Test Set:", mse)

# Plot the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


2----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create the Salary dataset
data = {'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]}

df = pd.DataFrame(data)

# Identify the independent and target variables
X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

# Split the variables into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Plot the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression Model')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


3----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv("D:/Computer Science/TY/SEM6/Slips/WT-2_DA-slips-solved/CSV/User_Data.csv")
# Select relevant features (Age and Estimated Salary) and the target variable (Purchased)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# Build a logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
# Make predictions on the testing set
y_pred = logistic_regression.predict(x_test)
# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# Plot the confusion matrix using seaborn heatmap
sn.heatmap(confusion_matrix, annot=True)
# Print the accuracy score
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
# Display the plot
plt.show()
# Print the testing set and predicted values
print(x_test)
print(y_pred)


4----
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the csv data
df = pd.read_csv("D:/Computer Science/TY/SEM6/Slips/WT-2_DA-slips-solved/CSV/fish.csv")

# Define feature columns 
feature_cols = ['Length', 'Diagonal', 'Height', 'Width']
X = df[feature_cols]  

# Target variable 
y = df['Weight']   

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create a LinearRegression and fit on training data
model = LinearRegression()
model.fit(X_train, y_train)   

# Evaluate model performance
print(f"R^2 Score on Train set: {model.score(X_train, y_train):.3f}")
print(f"R^2 Score on Test set: {model.score(X_test, y_test):.3f}")

# Visualization
plt.scatter(y_train, model.predict(X_train))
plt.xlabel("True Values")
plt.ylabel("Predictions") 

plt.show()


5----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\Computer Science\TY\SEM6\Data-Analytics\DA-pratical\CSV\iris.csv")

df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    print(f"\nStatistics for Species {species}:\n{species_data.describe()}")

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

6----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [['Bread','Milk'],
                ['Bread','Diaper','Beer','Eggs'],
                ['Milk','Diaper','Beer','coke'],
                ['Bread','Milk','Diaper','coke'],
                ['Bread','Milk','Diaper','Eggs']]
print(transactions)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)


7----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Read the dataset, specifying data types
df = pd.read_csv("C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/market.csv", dtype={'Transaction_ID': str, 'Items': str})
print("Columns:", df.columns)
# Drop null values
df.dropna(inplace=True)

# Convert items to list of lists
transactions = df['Items'].str.split(',')

# Convert categorical values to numeric using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Display information
print("Original Dataset:")
print(df.head())

# Generate frequent itemsets using Apriori algorithm with lower min_support
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

# Filter out itemsets with very low support before applying association rules
frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= 0.01]

# Generate association rules from frequent itemsets with lower min_threshold
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)

# Display information
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)


8----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Read the dataset, specifying data types
df = pd.read_csv("C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/market.csv", dtype={'Transaction_ID': str, 'Items': str})
# Drop null values
df.dropna(inplace=True)
# Convert items to list of lists
transactions = df['Items'].str.split(',')
# Convert categorical values to numeric using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
# Display information
print("Original Dataset:")
print(df.head())
# Generate frequent itemsets using Apriori algorithm with lower min_support
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
# Filter out itemsets with very low support before applying association rules
frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= 0.01]
# Generate association rules from frequent itemsets with lower min_threshold
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
# Display information
print("\nFrequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)


9----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = [['Austin','Kunal'],
                ['Austin','Naufil','Kunal'],
                ['Kunal','arpita','Beer','Naufil'],
                ['arpita','Austin','kunal','ishika'],
                ['Austin','arpita','Beer','Naufil']]
print(transactions)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)


10----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [['eggs','Milk','bread'],
                ['eggs','apple'],
                ['Milk','bread'],
                ['apple','Milk'],
                ['Milk','apple','bread']]
print(transactions)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)


11----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [['butter','bread','Milk'],
                ['butter','flour','Milk','sugar'],
                ['butter','eggs','Milk','salt'],
                ['eggs'],
                ['butter','flour','Milk','salt']]
print(transactions)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)


12----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create a random dataset with 10 samples
heights = np.random.normal(170, 10, 10)
weights = np.random.normal(70, 5, 10)
# Combine the two arrays into a single dataset
dataset = pd.DataFrame({'Height': heights, 'Weight': weights})

X_train, X_test, y_train, y_test = train_test_split(dataset[['Height']], dataset['Weight'], test_size=0.3, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Model Coefficients:', lr_model.coef_)
y_pred = lr_model.predict(X_test)
print('Predictions:', y_pred)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression Model')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()


13----
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Assuming your CSV file is named 'nursery_data.csv'
file_path = "C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/nursery-data.csv"
names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
dataset = pd.read_csv(file_path, names=names)

# Identify independent and target variables
X = pd.get_dummies(dataset.drop('class', axis=1))
Y = dataset['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)


14----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [['Apple','Mango','Banana'],
                ['Mango','Banana','Cabbage','Carrots'],
                ['Mango','Banana','Carrots'],
                ['Mango','Carrots']]
print("Transactions: \n",transactions)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print("Data Set: \n",df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print("Frequency Items: \n",freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print("Rules: \n",rules)


15----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = {'No': [1, 2, 3, 4],
                'Company': ['Tata', 'MG', 'Kia', 'Hyundai'],
                'Model': ['Nexon', 'Astor', 'Seltos', 'Creta'],
                'Year': ['2017', '2021', '2019', '2015']}

print("Transactions: \n", transactions)

# Convert transactions to a list of lists
transaction_list = [[str(transactions[key][i]) for key in transactions.keys()] for i in range(len(transactions['No']))]

te = TransactionEncoder()
te_array = te.fit(transaction_list).transform(transaction_list)
df = pd.DataFrame(te_array, columns=te.columns_)
print("Data Set: \n", df)

freq_items = apriori(df, min_support=0.1, use_colnames=True)
print("Frequency Items: \n", freq_items)

rules = association_rules(freq_items, metric='support', min_threshold=0.5)
rules = rules.sort_values(['support', 'confidence'], ascending=[False, False])
print("Rules: \n", rules)


16----
import re

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# Text to preprocess and summarize
text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human languages. It involves how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition and natural language generation. The history of natural language understanding and natural language processing generally started in the 1950s, although work can be found from earlier periods."

# Preprocess the text to remove special characters and digits
processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Tokenize the processed text
tokens = word_tokenize(processed_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Calculate word frequency
freq_dist = FreqDist(filtered_tokens)

# Generate summary using extractive summarization by selecting the most frequent words
summary_sentences = []
for sentence in sent_tokenize(text):
    sentence_tokens = word_tokenize(sentence)
    sentence_score = sum([freq_dist[token] for token in sentence_tokens])
    summary_sentences.append((sentence, sentence_score))

# Sort sentences by score and select the top 2 for summary
summary_sentences.sort(key=lambda x: x[1], reverse=True)
summary = ' '.join([sentence[0] for sentence in summary_sentences[:2]])

print(summary)


17----
import re

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# Text to preprocess and summarize
text = "Consider text paragraph. So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work."

# Preprocess the text to remove special characters and digits
processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Tokenize the processed text
tokens = word_tokenize(processed_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Calculate word frequency
freq_dist = FreqDist(filtered_tokens)

# Generate summary using extractive summarization by selecting the most frequent words
summary_sentences = []
for sentence in sent_tokenize(text):
    sentence_tokens = word_tokenize(sentence)
    sentence_score = sum([freq_dist[token] for token in sentence_tokens])
    summary_sentences.append((sentence, sentence_score))

# Sort sentences by score and select the top 2 for summary
summary_sentences.sort(key=lambda x: x[1], reverse=True)
summary = ' '.join([sentence[0] for sentence in summary_sentences[:2]])

print(summary)


18----
import re

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Text paragraph to process
text = "Consider any text paragraph. Remove the stopwords. Tokenize the paragraph to extract words and sentences. Calculate the word frequency distribution and plot the frequencies. Plot the wordcloud of the text."

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Calculate word frequency distribution
freq_dist = FreqDist(filtered_tokens)

# Plot word frequency distribution
plt.figure(figsize=(12, 6))
freq_dist.plot(20, cumulative=False)

# Generate and display word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of the Text')
plt.show()


19----
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud

# Load the dataset
df = pd.read_csv("C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/movie_review.csv")

# Add a column for sentiment analysis using TextBlob
df['Sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Create a new dataframe for positive reviews only
pos_df = df[df['Sentiment'] > 0.2]

# Create a word cloud for positive reviews
wordcloud = WordCloud(width=500, height=500, background_color='white', stopwords=STOPWORDS, min_font_size=8).generate(' '.join(pos_df['text']))

# Plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


20----
import nltk
from nltk.corpus import stopwords

# Text paragraph
text = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word.lower() in stop_words]

# Print the filtered tokens
print(filtered_tokens)


21----
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Collect data
data = pd.read_csv("C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/User_Data.csv")

# 2. Preprocess data
data.dropna(inplace=True)
X = data['Age'].values.reshape(-1, 1)
Y = data['Salary'].values.reshape(-1, 1)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 5. Predict values
y_pred = regressor.predict(X_test)

# 6. Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error:", mse)
print("R squared:", r2)

# 7. Visualize results
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


22----
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample text paragraph
text = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."

# Tokenize the text paragraph
words = word_tokenize(text)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [word for word in words if word.casefold() not in stop_words]

# Join filtered words to form a sentence
filtered_sentence = ' '.join(filtered_words)

print(filtered_sentence)


23----
import re

# Sample text paragraph
text = "Hello, #world123! This is a sample text paragraph. It contains special characters and digits."

# Remove special characters and digits
processed_text = re.sub(r'[^a-zA-Z\s]', '', text)

print(processed_text)


24----
import pandas as pd

# Read the dataset
df = pd.read_csv("C:/xampp/htdocs/WT-2/WT-2_DA-slips-solved/CSV/INvideos.csv")

# Drop the columns that are not required
df = df.drop(['video_id', 'trending_date', 'channel_title', 'category_id', 'publish_time', 'tags',
              'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed'], axis=1)

# Convert the datatype of 'views', 'likes', 'dislikes', and 'comment_count' to integer
df[['views', 'likes', 'dislikes', 'comment_count']] = df[['views', 'likes', 'dislikes',
                                                          'comment_count']].astype(int)

# Find the total views, likes, dislikes, and comment count
total_views = df['views'].sum()
total_likes = df['likes'].sum()
total_dislikes = df['dislikes'].sum()
total_comments = df['comment_count'].sum()

print('Total Views:', total_views)
print('Total Likes:', total_likes)
print('Total Dislikes:', total_dislikes)
print('Total Comments:', total_comments)


26----
import re

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text to summarize
text = "Hello all, #Welcome to @Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."

# Preprocess the text to remove special characters and digits
preprocessed_text = re.sub(r'[^a-zA-Z\s]', '', text)

# Tokenize the preprocessed text into sentences
sentences = sent_tokenize(preprocessed_text)

# Calculate the importance score of each sentence using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)
cosine_similarity_matrix = cosine_similarity(tfidf_matrix)

# Select top N sentences based on their importance score
N = 2
top_sentences = sorted(range(len(cosine_similarity_matrix[-1])), key=lambda i: cosine_similarity_matrix[-1][i])

# Concatenate the top sentences to form the summary
summary = ''
for i in top_sentences[:N]:
    summary += sentences[i] + ' '

print(summary)


27----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = [['Austin','Kunal'],
                ['Austin','Naufil','Kunal'],
                ['Kunal','arpita','Beer','Naufil'],
                ['arpita','Austin','kunal','ishika'],
                ['Austin','arpita','Beer','Naufil']]
print(transactions)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)


28----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the car dataset
data = {'Mileage': [10, 20, 30, 40, 50],
        'Price': [24, 19, 17, 13, 10]
        }

# Convert the data dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Select relevant features (Mileage) and the target variable (Price)
X = df[['Mileage']].values
y = df['Price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Build a linear regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = linear_regression.predict(X_test)

# Print the model performance metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression - Car Dataset')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()


29----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create a simple student score dataset
data = {'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1]}
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df[['Hours_Studied']]
y = df['Pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Build a logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logistic_regression.predict(X_test)

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

# Plot the confusion matrix using seaborn heatmap
sn.heatmap(confusion_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.show()

# Print the accuracy score
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Testing Set:\n',X_test)
print('Predicted Values:\n',y_pred)


30----
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = [['eggs','Milk','bread'],
                ['eggs','apple'],
                ['Milk','bread'],
                ['apple','Milk'],
                ['Milk','apple','bread']]
print(transactions)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
freq_items = apriori(df, min_support=0.5,use_colnames = True)
print(freq_items)
rules = association_rules(freq_items,metric='support',min_threshold=0.5)
rules = rules.sort_values(['support','confidence'], ascending=[False,False])
print(rules)

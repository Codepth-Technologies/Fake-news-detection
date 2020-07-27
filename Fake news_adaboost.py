import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn.svm import SVC
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

real = "C:/Users/KIIT/Documents/xanxit/datasets_72366_159129_BuzzFeed_fake_news_content.csv"
fake = "C:/Users/KIIT/Documents/xanxit/datasets_72366_159129_BuzzFeed_fake_news_content.csv"
df = pd.read_csv(fake)
dff = pd.read_csv(real)
df.head()
df['label'] = 'fake'
df = df[['title', 'text', 'url', 'authors', 'source', 'meta_data', 'label']]
df.head()
dff.head()
dff['label'] = 'real'
df = df[['title', 'text', 'url', 'authors', 'source', 'meta_data', 'label']]
dff = dff[['title', 'text', 'url', 'authors', 'source', 'meta_data', 'label']]
df.shape
dff.shape
df = pd.concat([df, dff])
df.shape
df = df.sample(frac = 1)

df.head()
df.columns
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

df['title'] = df['title'].apply(tweet_cleaner)
df['text'] = df['text'].apply(tweet_cleaner)
df['meta_data'] = df['meta_data'].apply(tweet_cleaner)
def all_X(row):
    try:
         return row['title']+" "+row["text"]+" "+row["meta_data"]
            #giving out the error fields.
    except:
           print("error", row)

df["all_X"]= df.apply(all_X,axis=1)
df = df[['all_X', 'label']]
df.columns
X = df.all_X
y = df.label
count_vec = CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df = 3) 
                           
#Fit Count Vectorizer
dtm_cv = count_vec.fit_transform(X)
#Convert it to a pandas data frame
df_cv = pd.DataFrame(dtm_cv.toarray(), columns=count_vec.get_feature_names())
df_cv.head()
df_cv.shape
x_train, x_test, y_train, y_test = train_test_split(df_cv,y,test_size=0.1)
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(x_train, y_train)
# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)
# print classification report 
grid_predictions = grid.predict(x_test) 
print(classification_report(y_test, grid_predictions))
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
accuracy = cross_val_score(estimator = dt, X = x_train, y = y_train, cv=10)
print("Model accuracy : {:0.2f}%".format(accuracy_score(y_pred,y_test)*100))
print("cross validation : {:0.2f}%".format(accuracy.mean()*100))

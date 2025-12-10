import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv('phishing_site_urls.csv')

# Tokenize URLs
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
df["text_tokenized"] = df.URL.map(lambda x: tokenizer.tokenize(x))

# Stem tokens
stemmer = SnowballStemmer("english")
df["text_stemmed"] = df["text_tokenized"].map(lambda l: [stemmer.stem(y) for y in l])

# Join stemmed tokens into single string
df["text"] = df["text_stemmed"].map(lambda l: " ".join(l))

# Separate good and bad sites
good_sites = df[df.Label == 'good']
bad_sites = df[df.Label == 'bad']

# Wordcloud plot function
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'http','https','www','com','net','org'}
    stopwords = stopwords.union(more_stopwords)
    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          mask=mask)
    wordcloud.generate(text)

    plt.figure(figsize=figure_size)
    if image_color and mask is not None:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
        plt.title(title, fontdict={'size':title_size, 'verticalalignment':'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size':title_size,'color':'green' ,'verticalalignment':'bottom'})

    plt.axis('off')
    plt.tight_layout()
    #plt.show()

# Generate wordclouds for good and bad sites
all_text_good = " ".join(good_sites["text"].tolist())
wordcloud_good = WordCloud(background_color='white', height=400, width=800).generate(all_text_good)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_good, interpolation='bilinear')
plt.axis('off')
#plt.show()

all_text_bad = " ".join(bad_sites["text"].tolist())
wordcloud_bad = WordCloud(background_color='white', height=400, width=800).generate(all_text_bad)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_bad, interpolation='bilinear')
plt.axis('off')
#plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
# Vectorize text
cv = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.8)
features = cv.fit_transform(df.text)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(features, df.Label, test_size=0.2, random_state=42)

# Logistic Regression model training
lr_model = LogisticRegression(max_iter=200, solver='liblinear')
lr_model.fit(x_train, y_train)

print("Logistic Regression Test Accuracy:", lr_model.score(x_test, y_test))
print("Logistic Regression Train Accuracy:", lr_model.score(x_train, y_train))

# Confusion matrix plot
con_mat = pd.DataFrame(
    confusion_matrix(y_test, lr_model.predict(x_test)),
    index=['Actual: bad','Actual: good'],
    columns=['Predicted: bad','Predicted: good']
)

plt.figure(figsize=(6,4))
sns.heatmap(con_mat, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Logistic Regression Confusion Matrix")
#plt.show()

# Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
print("Multinomial Naive Bayes Test Accuracy:", mnb.score(x_test, y_test))

# Save models and vectorizer with pickle
pickle.dump(lr_model, open('phishing.pkl', 'wb'))
pickle.dump(mnb, open('phishing_mnb.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

# -------- SHAP Explanation --------
# -------- SHAP Explanation (SAFE & CLEAR BAR PLOT) --------
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

# IMPORTANT: Manually set class order (same as training)
lr_model.classes_ = np.array(['bad', 'good'])

# Small safe background sample (avoid memory crash)
background_urls = [
    "http://example.com",
    "https://google.com",
    "https://github.com",
]
background_vec = cv.transform(background_urls)

# Dense mean vector
mean_dense = background_vec.mean(axis=0).A1  # 1D array

# Model coefficients
coef_dense = lr_model.coef_.reshape(-1)

# Select one test example to explain
sample_index = 0
input_dense = x_test[sample_index].toarray().reshape(-1)

# Compute SHAP values manually (linear formula)
shap_values = (input_dense - mean_dense) * coef_dense

# Top contributing features
top_n = 12
feature_names = cv.get_feature_names_out()

indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
top_features = feature_names[indices]
top_importance = shap_values[indices]

# ---------------- Draw Explanation Bar Plot ----------------
plt.figure(figsize=(10, 5))
colors = ['red' if val > 0 else 'blue' for val in top_importance]
plt.barh(top_features, top_importance, color=colors)

plt.title("Top URL Features Influencing Prediction")
plt.xlabel("Contribution to Phishing vs Legitimate")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save for web UI OR local preview
if not os.path.exists("static"):
    os.mkdir("static")

plt.savefig("static/shap_bar.png")  
plt.close()

print("\n✔ SHAP Bar Explanation Generated Successfully!")

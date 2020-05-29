
import pandas as pd
import numpy as np
from sklearn import datasets
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import operator
import spacy
nlp = spacy.load("en_core_web_sm")


class ReziewAlgorithm:

    data = None
    reviews_ratings = None
    all_reviews = None
    review_train = None
    review_test = None
    rating_train = None
    rating_test = None

    vec_review_train = None
    vec_review_test = None

    stop_words = ['a', 'the', 'that', 'of', 'which', 'i', 'it',
                  'this', 'too', 'sometime', 'sometimes', 'slightly', 'somewhat']
    cv = CountVectorizer(binary=True, ngram_range=(1, 3),
                         stop_words=stop_words)

    max_c = None
    final_model = None

    sorted_words = None
    word_score_dict = {}
    word_counts = {}

    # read in the .csv file and drops all rows without a valid review or rating
    def read_csv(self, file, review_col, rating_col):
        data = pd.read_csv(file)
        data.drop(data[data['lang'] != 'en'].index, inplace=True)
        data[rating_col].replace(r"\D", "", inplace=True, regex=True)
        data[rating_col].replace("", np.nan, inplace=True)
        data[review_col].replace("", np.nan, inplace=True)
        data.dropna(subset=[rating_col], inplace=True)
        data.dropna(subset=[review_col], inplace=True)
        self.data = data

    def __init__(self, file, review_col, rating_col):
        self.read_csv(file, review_col, rating_col)

    # store the review and rating columns as a class variable
    def extract_cols(self):
        reviews = pd.DataFrame()
        reviews['reviews'] = self.data['review_text']
        reviews['ratings'] = self.data['grade']
        self.reviews_ratings = reviews

    # convert ratings to binary scale and split into train and test subset
    def preprocess_ratings(self):
        ratings = self.reviews_ratings['ratings']
        n = int(len(ratings) * (2/3))
        rating_list = ratings.tolist()

        def convert_rating(x):
            x = int(x)
            if x < 4:
                return 0
            else:
                return 1
        rating_list = [convert_rating(x) for x in rating_list]
        rating_train = rating_list[0:n]
        rating_test = rating_list[n:]
        self.rating_train = rating_train
        self.rating_test = rating_test

    # remove punctuation from reviews and split into train and test subset
    def preprocess_reviews(self):
        reviews = self.reviews_ratings['reviews']
        n = int(len(reviews) * (2/3))
        review_list = reviews.tolist()
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(\-)|(\/)")
        processed_reviews = [REPLACE_NO_SPACE.sub(
            "", str(line).lower()) for line in review_list]
        processed_reviews = [REPLACE_WITH_SPACE.sub(
            " ", line) for line in review_list]
        reviews_train = processed_reviews[0:n]
        reviews_test = processed_reviews[n:]
        self.all_reviews = processed_reviews
        self.review_train = reviews_train
        self.review_test = reviews_test

    # transform reviews into vector input for regression model
    def vectorize_reviews(self):
        self.cv.fit(self.review_train)
        print(self.cv)
        self.vec_review_train = self.cv.transform(self.review_train)
        print(self.vec_review_train)
        self.vec_review_test = self.cv.transform(self.review_test)

    # train logistic regression model on different parameters and returns parameter with highest accuracy
    # returns a c value as a coefficient to build logistic regression model
    def train_model_for_c(self):
        X_train, X_test_values, y_train, Y_test_values = train_test_split(
            self.vec_review_train, self.rating_train, train_size=0.75
        )

        c_list = np.linspace(1, 1.3, num=3)
        c_accuracy = {}
        for c in c_list:
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            c_accuracy[c] = accuracy_score(
                Y_test_values, lr.predict(X_test_values))
            print("Accuracy for C=%s: %s"
                  % (c, accuracy_score(Y_test_values, lr.predict(X_test_values))))
        self.max_c = max(c_accuracy.items(), key=operator.itemgetter(1))[0]

    # fit a model using the optimal parameter
    def get_final_model(self):
        final_model = LogisticRegression(C=self.max_c)
        self.final_model = final_model.fit(
            self.vec_review_train, self.rating_train)

    # test optimal model on test data subset and print accuracy
    def test_model(self):
        print("Final Accuracy: %s"
              % accuracy_score(self.rating_test, self.final_model.predict(self.vec_review_test)))

    # store dictionary of words and sentiment scores as class variable
    def get_words_and_scores(self):
        feature_to_coef = {
            word: coef for word, coef in zip(
                self.cv.get_feature_names(), self.final_model.coef_[0]
            )
        }

        self.sorted_words = sorted(feature_to_coef.items(), key=lambda x: x[1])
        self.word_score_dict = dict(self.sorted_words)

    def get_most_positive_negative_words(self):
        num_words = 10
        word_dict = {'positive': {'words': [], 'weights': [], 'reviews': [], 'hits': []},
                     'negative': {'words': [], 'weights': [], 'reviews': [], 'hits': []}}

        for best_positive in reversed(self.sorted_words[-num_words:]):
            positive_dict = word_dict['positive']
            positive_dict['words'].append(best_positive[0])
            positive_dict['weights'].append(best_positive[1])
            print(best_positive)

        for best_negative in self.sorted_words[:num_words]:
            negative_dict = word_dict['negative']
            negative_dict['words'].append(best_negative[0])
            negative_dict['weights'].append(best_negative[1])
            print(best_negative)

    # return sentiment score of a given word, or None if that word does not exist in the corpus
    def get_sentiment_score(self, word):
        word = word.strip().lower()
        if word in self.word_score_dict:
            return self.word_score_dict[word]
        else:
            return None

    # return the number of reviews in which a given word appears
    def get_review_count(self, word):
        if word in self.word_counts:
            return self.word_counts[word]
        hit_list = [idx for idx, s in enumerate(self.all_reviews) if word in s]
        self.word_counts[word] = len(hit_list)
        return self.word_counts[word]

    # helper function for analyze(). finds all of the adjs associated with a noun
    def recurse(self, adj):
        word = adj.text
        if self.get_sentiment_score(word) == None or self.get_review_count(word) == 0:
            return []
        out = [(word, self.get_sentiment_score(
            word), self.get_review_count(word))]
        for child in adj.children:
            if child.pos_ == "ADJ":
                out += self.recurse(child)
        return out

    # reutrn a dictionary of nouns and the adjectives that are associated with them
    def analyze(self, s):
        s = s.lower()
        doc = nlp(s)
        pairs = {}
        for token in doc:
            if token.pos_ == "AUX":
                n = ""
                a = []
                for child in token.children:
                    if child.pos_ == "NOUN" and len(child.text) > 2:
                        n = child.text
                    elif child.pos_ == "ADJ":
                        a += self.recurse(child)
                if n != "" and a != []:
                    if n in pairs:
                        pairs[n] = sorted(set(pairs[n] + a))
                    else:
                        pairs[n] = sorted(set(a))
            if token.pos_ == "NOUN" and len(token.text) > 2:
                a = []
                for child in token.children:
                    if child.pos_ == "ADJ":
                        # Recurse might not be neccesary
                        a += self.recurse(child)
                if a != []:
                    if token.text in pairs:
                        pairs[token.text] = sorted(set(pairs[token.text] + a))
                    else:
                        pairs[token.text] = sorted(set(a))
        return pairs

    # shortcut function for training model on dataset
    def train_model(self):
        self.extract_cols()
        self.preprocess_ratings()
        self.preprocess_reviews()
        self.vectorize_reviews()
        self.train_model_for_c()
        self.get_final_model()


model = ReziewAlgorithm('reziew.csv', 'review_text', 'grade')
model.train_model()

model.reviews_ratings.head()

print(model.max_c)

model.test_model()
model.get_words_and_scores()
model.get_most_positive_negative_words()

long_review = ''
revs = model.all_reviews
for i in range(0, len(revs)):
    if len(long_review) < 999000:
        try:
            long_review += (revs[i] + '. ')
        except:
            pass
#rev = reviews['reviews'][4]
# print(long_review)
result = model.analyze(long_review)
print(result)

json_list = []
for k, v in result.items():
    ct = model.get_review_count(k)
    if (ct > 0):
        new_dict = {}
        new_dict['noun'] = k
        new_dict['noun_count'] = ct
        new_dict['adjs'] = [tup[0] for tup in v]
        new_dict['adjs_score'] = [tup[1] for tup in v]
        new_dict['adjs_count'] = [tup[2] for tup in v]
        new_dict['noun_score'] = (np.sum(np.multiply(
            new_dict['adjs_score'], new_dict['adjs_count'])))/np.sum(new_dict['adjs_count'])
        json_list.append(new_dict)


def get_noun_count(json):
    try:
        return int(json['noun_count'])
    except KeyError:
        return 0


def get_noun_score(json):
    try:
        return int(json['noun_score'])
    except KeyError:
        return 0


json_list.sort(key=get_noun_score, reverse=True)
json_output = json.dumps(json_list)
print(json_output)

print(model.word_score_dict)

# LEFTOVER CODE FOR GETTING REVIEW IN WHICH A WORD APPEARS
# for word in word_dict['positive']['words']:
#     hit_list = [idx for idx, s in enumerate(reviews_clean) if word in s]
#     first_hit = hit_list[0]
#     word_dict['positive']['reviews'].append(review_list[first_hit])
#     word_dict['positive']['hits'].append(len(hit_list))
#     print(review_list[first_hit])
#     print(len(hit_list))
#     print("\n")

# for word in word_dict['negative']['words']:
#     hit_list = [idx for idx, s in enumerate(reviews_clean) if word in s]
#     first_hit = hit_list[0]
#     word_dict['negative']['reviews'].append(review_list[first_hit])
#     word_dict['negative']['hits'].append(len(hit_list))
#     print(review_list[first_hit])
#     print(len(hit_list))
#     print("\n")

import pickle
from operator import itemgetter
from langdetect import detect_langs
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import classification_report

SEGMENT_NOT_SUPPORTED = "Not supported language"

def isEnglish(s):
    """
    Detect if a sentence is written in English
    :param s: sentence string
    :return: boolean - True for English
    """
    try:
        detection = detect_langs(s[:50])
        lang_obj = detection[0]
        return lang_obj.lang == 'en' and lang_obj.prob > 0.95
    except:
        return False


class Classifier:
    def __init__(self, dataset_json=None, model_file_name=None):
        """
        Constructor
        :param dataset_file_name: An Excel file to train the classifier with
        :param model_file_name: A pickled file to hold a pre-trained model
        """
        if dataset_json is not None:
            # try to train on a json dataset
            self.train(dataset_json)
        elif model_file_name is not None:
            # try to load pre-trained model
            try:
                with open(model_file_name, 'r') as f:
                    self.clf, self.vectorizer = pickle.load(f)
                    print ('Loaded model from file {}'.format(model_file_name))
            except:
                print ('Unable to load model from file {}'.format(model_file_name))

    def get_classes(self):
        return self.clf.classes_.tolist()

    def predict_one (self, string):
        """
        Predict top segments for one input
        :param string: input
        :return: list of sorted segments with probabilities
        """
        if not isEnglish(string):
            return [[SEGMENT_NOT_SUPPORTED, 1.00]]

        _input = self.vectorizer.transform([string])
        predictions = (zip(self.get_classes(), self.clf.predict_proba(_input)[0]))
        return sorted(predictions, key=itemgetter(1), reverse=True)

    def predict_batch (self, batch):
        """
        Predict top segment for each in a batch
        :param batch: list of inputs
        :return: predictions in the same order of the inputs
        """
        # get the predictions for the entire batch at once
        batch_vec = self.vectorizer.transform(batch)
        predictions = self.clf.predict(batch_vec).tolist()

        # override non-english strings in the batch
        for i, s in enumerate(batch):
            if not isEnglish(s):
                predictions[i] = SEGMENT_NOT_SUPPORTED

        return predictions

    def train (self, dataset_json, output_model_file_name='model.pcl'):
        """
        :param dataset_json:
        :param output_model_file_name:
        :return:
        """
        print ('Starting to train the model...')
        # load the dataset into pandas
        df = pd.read_json(json.dumps(dataset_json))
        # use the english descriptions only
        df2 = df[df['description'].apply(isEnglish)]
        # remove segments that we dont have enough data for
        # I choose not to predict those segments if I don't
        # have enough data...
        sements_counts = df2['segment'].value_counts()
        total_counts = df2.shape[0]
        thres = int(.03 * total_counts) # threshold?
        sements_counts = sements_counts[sements_counts > thres] # allow above threshold
        allowed_sements = sements_counts.index.tolist() # array of our segments

        df2 = df2.loc[df2['segment'].isin(allowed_sements)] # keep rows only from those segments

        # split to training and testing data
        X_train, X_test, y_train, y_test = train_test_split(df2, df2['segment'], test_size=.10)
        # tf-idf vectorization of our corpus

        self.vectorizer = TfidfVectorizer(
            use_idf=True,
            norm=None,
            smooth_idf=True,
            sublinear_tf=False,
            binary=False,
            min_df=1,
            max_df=.2,  # ignore very commmon terms
            max_features=None,
            strip_accents=None,
            ngram_range=(1, 1),
            preprocessor=None,
            stop_words=ENGLISH_STOP_WORDS,
            tokenizer=None,
            vocabulary=None
        )
        counts = self.vectorizer.fit_transform(X_train['description'].values)

        self.clf = MultinomialNB()
        targets = y_train.values
        self.clf.fit(counts, targets)

        # pickle the model
        with open(output_model_file_name, "wb") as f:
            pickle.dump((self.clf, self.vectorizer), f)

        X_test_tokened = self.vectorizer.transform(X_test['description'].values)
        predicted = self.clf.predict(X_test_tokened)

        print('Accuracy is {}'.format(accuracy_score(y_test.values, predicted)))

        # tp / (tp + fp)
        # out of the answers we gave to the user - what was actually true?
        print('Precision is {}'.format(precision_score(y_test.values, predicted, average='weighted')))

        # tp / (tp + fn)
        # how much gold did we dig out of the good stuff?
        # how much do the app creators agree with our classifier?
        print('Recall is {}'.format(recall_score(y_test.values, predicted, average='weighted')))

        print('Classification report on test set for classifier:')
        print(classification_report(y_test.values, predicted,
                                    target_names=self.clf.classes_))

        print('Trained model!')
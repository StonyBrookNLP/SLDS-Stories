import sys
import csv
import spacy
import data_utils as du
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

trn_path = sys.argv[1]
val_path = sys.argv[2]

out_vocab = du.sentiment_label_vocab()
target_names = out_vocab.itos
print(target_names)

# tokenizer
nlp = spacy.load('en')

# prepare the dataset
def get_data(path):
    all_data, all_target = [], []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # skip the header
        for line in csv_reader:
            sents = line[2:7] 
            targets = line[-5:]
            for idx, sent in enumerate(sents):
                tokens = nlp.tokenizer(sent)
                data = " ".join([x.text for x in tokens])
                target = targets[idx]
                # each sentence is a sample
                all_data.append(data)
                all_target.append(target_names.index(target))

    return all_data, all_target

data_train, target_train = get_data(trn_path)

# prepare input features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data_train)
y_train = np.array(target_train)
print("X_train shape {}".format(X_train.shape))
print("y_train shape {}".format(y_train.shape))

# create a classifier
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

print(clf)
print("Training accuracy {:.4f}".format(clf.score(X_train, y_train)))

# validation
data_valid, target_valid = get_data(val_path)
X_valid = vectorizer.transform(data_valid)
y_valid = np.array(target_valid)
preds = clf.predict(X_valid)
score = metrics.accuracy_score(y_valid, preds)
print("Validation accuracy: {:.4f}".format(score))

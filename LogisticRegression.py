import numpy as np
import pandas as pd
import sklearn.linear_model as sk
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("train.csv") #reads in data
results = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] #the outcomes
text = data["comment_text"] #stores in the actual text
categories = list(results)  #takes the categories above and stores it in list
#print(categories)

d = []
length = len(text) // 100
results = results[:length]

# DOES NECESSARY PARSING AND CONVERTING TEXT TO NUMERICAL DATA (Bag of Words)
for i in range(length):
    words = text[i]
    content = words.split()
    counts = {x:words.count(x) for x in set(content)}
    #print(counts)
    #counts['id'] = data["id"][i]
    d = d + [counts]

df = pd.DataFrame(d).fillna(0)

# SPLITS DATASET TO TRAINING SET AND VALIDATION SET
keep = np.random.rand(len(df)) < 0.8
X_train = df[keep]
Y_train = results[keep]
X_valid = df[~keep]
Y_valid = results[~keep]

def multi_evaluate():
    print("USING LOGISTIC REGRESSION:")

	# TRAINING AND EVALUATING ON VALIDATION SET
    for cat in categories:
        log_reg_toxic = sk.LogisticRegression()
        log_reg_toxic.fit(X_train, Y_train[cat])
        valid_score = log_reg_toxic.score(X_valid, Y_valid[cat])
        print("For category", cat)
        print("The validation error is:", valid_score)


def multi_validation():
    penalty = ["l1", "l2"]
    C = [0.1, 0.3, 1, 3, 10]
    max_iter = [100, 125, 150, 175, 200]

    param = {'penalty' : penalty, 'C' : C, "max_iter" : max_iter}

    for cat in categories:
        model = sk.LogisticRegression()
        clf = GridSearchCV(model, param)
        clf.fit(df, results[cat])
        print("For category", cat, " the best parameters are")
        print(clf.best_estimator_)
        print("With a validation score of", clf.best_score_)

    return None

multi_validation()

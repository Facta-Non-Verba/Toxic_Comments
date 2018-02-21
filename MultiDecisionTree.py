import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("train.csv") #reads in data
results = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] #the outcomes
text = data["comment_text"] #stores in the actual text
categories = list(results) #takes the categories above and stores it in list
#print(categories)

d = []
length = len(text) // 100
results = results[:length]

# DOES NECESSARY PARSING AND CONVERTING TEXT TO NUMERICAL DATA (Bag of Words)
for i in range(length):
    words = text[i]
    content = words.split()
    counts = {x:words.count(x) for x in words}
    counts['id'] = data["id"][i]
    d.append(counts)

df = pd.DataFrame(d).fillna(0)._get_numeric_data()

# SPLITS DATASET TO TRAINING SET AND VALIDATION SET
keep = np.random.rand(len(df)) < 0.8
X_train = df[keep]
Y_train = results[keep].astype(int)
X_valid = df[~keep]
Y_valid = results[~keep].astype(int)

#print(Y_train)


def validation():
    #Validation for selecting hyper-parameters
    print("Beginning validation to chose parameters")
    num_trees = [5, 10, 20]
    depth = [None] + [5, 15]
    split = [2, 5, 8]
    samples_leaf = [1, 3, 5]

    scores = {}
    params = {}
    for cat in categories:
        scores[cat] = float("inf")

    for n_trees in num_trees:
        for d in depth:
            for s in split:
                for samples in samples_leaf:
                    print("Now considering parameters:", (n_trees, d, s, samples))
                    for cat in categories:
                        model = RandomForestClassifier(n_estimators = n_trees, max_depth = d,
                                                    min_samples_split = s, min_samples_leaf = samples)
                        model.fit(X_train, Y_train[cat])

                        valid_score = model.score(X_valid, Y_valid[cat])
                        if scores[cat] > valid_score:
                            scores[cat] = valid_score
                            params[cat] = (n_trees, d, s, samples)

    for cat in categories:
        print("The optimal hyperparameters for", cat, " is", params[cat])
        print("The validation score associated was", scores[cat])



# SLIGHTLY MISNAMED, USES RANDOM FOREST
def main():
    for cat in categories:
        model = RandomForestClassifier()
        model.fit(X_train, Y_train[cat])

        train_score = model.score(X_train, Y_train[cat])
        valid_score = model.score(X_valid, Y_valid[cat])
        print("The training error for the", cat, "case is:", train_score)
        print("The training error for the validation case is:", valid_score)


validation()

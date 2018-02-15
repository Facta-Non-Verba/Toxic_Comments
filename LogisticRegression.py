import numpy as np
import pandas as pd
import sklearn.linear_model as sk

data = pd.read_csv("train.csv") #reads in data
results = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] #the outcomes
text = data["comment_text"] #stores in the actual text
categories = list(results) #takes the categories above and stores it in list
#print(categories)

d = []
length = len(text)
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
Y_train = results[keep]
X_valid = df[~keep]
Y_valid = results[~keep]

print("USING LOGISTIC REGRESSION:")

# TRAINING AND EVALUATING ON VALIDATION SET
for cat in categories:
    log_reg_toxic = sk.LogisticRegression()
    log_reg_toxic.fit(X_train, Y_train[cat])

    train_score = log_reg_toxic.score(X_train, Y_train[cat])
    valid_score = log_reg_toxic.score(X_valid, Y_valid[cat])
    print("The training error for the", cat, "case is:", train_score)
    print("The training error for the validation case is:", valid_score)

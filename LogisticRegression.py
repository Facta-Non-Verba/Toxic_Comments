import numpy as np
import pandas as pd
import sklearn.linear_model as sk

data = pd.read_csv("train.csv")
results = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
text = data["comment_text"]
categories = list(results)
#print(categories)

d = []
length = len(text) // 10
results = results[:length]

for i in range(length):
    words = text[i]
    content = words.split()
    counts = {x:words.count(x) for x in words}
    counts['id'] = data["id"][i]
    d.append(counts)

df = pd.DataFrame(d).fillna(0)._get_numeric_data()

keep = np.random.rand(len(df)) < 0.8
X_train = df[keep]
Y_train = results[keep]
X_valid = df[~keep]
Y_valid = results[~keep]

toxic_train = Y_train["toxic"]
severe_toxic_train = Y_train["severe_toxic"]
obscene_train = Y_train["obscene"]
threat_train = Y_train["threat"]
insult_train = Y_train["insult"]
identity_hate_train = Y_train["identity_hate"]

toxic_valid = Y_valid["toxic"]
severe_toxic_valid = Y_valid["severe_toxic"]
obscene_valid = Y_valid["obscene"]
threat_test = Y_valid["threat"]
insult_test = Y_valid["insult"]
identity_hate_test = Y_valid["identity_hate"]



print("USING LOGISTIC REGRESSION:")

### TOXIC CASE
for cat in categories:
    log_reg_toxic = sk.LogisticRegression()
    log_reg_toxic.fit(X_train, Y_train[cat])

    train_score = log_reg_toxic.score(X_train, Y_train[cat])
    valid_score = log_reg_toxic.score(X_valid, Y_valid[cat])
    print("The training error for the", cat, "case is:", train_score)
    print("The training error for the validation case is:", valid_score)

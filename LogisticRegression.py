import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import json
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


def multi_evaluate():
	print("USING LOGISTIC REGRESSION:")

	# TRAINING AND EVALUATING ON VALIDATION SET
	for cat in categories:
	    log_reg_toxic = sk.LogisticRegression()
	    log_reg_toxic.fit(X_train, Y_train[cat])

	    train_score = log_reg_toxic.score(X_train, Y_train[cat])
	    valid_score = log_reg_toxic.score(X_valid, Y_valid[cat])
	    print("The training error for the", cat, "case is:", train_score)
	    print("The training error for the validation case is:", valid_score)

def multi_validation():
	penalty = ['l1', 'l2']
	C = [0.1, 0.3, 1, 3, 10] #smaller means stronger regularization
	max_iter = [100, 125, 150, 175, 200]

	param = {'penalty':['l1', 'l2'], 'C': [0.1, 0.3, 1, 3, 10], 'max_iter': [100, 125, 150, 175, 200]}

	for cat in categories:
		name = "validation_multi_Log_reg_" + cat + ".txt"
		model = sk.LogisticRegression()
		clf = GridSearchCV(model, param)
		clf.fit(df, results[cat])

		test = pd.DataFrame(clf.cv_results_)
		test.to_csv(name, sep = "\t")



multi_validation()


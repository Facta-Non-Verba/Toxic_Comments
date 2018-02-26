import numpy as np
import pandas as pd

data = pd.read_csv("train.csv") #reads in data
results = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] #the outcomes
text = data["comment_text"] #stores in the actual text
categories = list(results)  #takes the categories above and stores it in list
#print(categories)

d = []
length = len(text) // 100
results = results[:length]

def parse_data():
    # DOES NECESSARY PARSING AND CONVERTING TEXT TO NUMERICAL DATA (Bag of Words)
    for i in range(length):
        words = text[i]
        content = words.split()
        counts = {x:words.count(x) for x in words}
        counts['id'] = data["id"][i]
        d.append(counts)

    df = pd.DataFrame(d).fillna(0)._get_numeric_data()
    name = "single_letter_all_entries.csv"
    df.to_csv(name)

def parse_results():
    name = "fraction_results.csv"
    results.to_csv(name)

parse_results()

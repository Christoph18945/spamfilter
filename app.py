#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""app.py

Spam filter with Multinomial Naives Bayes Algorithm (frequency, discrete values)
data can be found in: https://www.kaggle.com/uciml/sms-spam-collection-dataset

The data are ~5000 messages. Empty columsn were removed from
the file. The headers have been renamed.

Explanation of the data:
ham  - no spam message
spam - spam message

The accumulation of messages are displyed as a table:
Each sentence is one line in the table.
Each word of the sentence is one cell.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def main() -> None:
    """Main program and entrypoint."""

    # read in data
    df: pd.DataFrame = pd.read_csv(r"./spam.csv", encoding="utf-8")
    print(len(df))

    # get column "message"
    X: pd.Series = df["message"]
    # get "type" column
    y: pd.Series = df["type"]

    # Execute train/test split. Split data in test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # Problem: Naive Bayes can not dealt with words. It
    # needs numbers. Means words must be converted into
    # features.
    cv = CountVectorizer(min_df = 0.001, max_df = 0.25)

    # Learn the vocabulary
    cv.fit(X_train)

    # Apply on test and training data. Convert
    # this in a spars matrix. It is a nested numpy array
    # but in a different display on the computer.
    X_train = cv.transform(X_train)
    X_test = cv.transform(X_test)

    # The (Multinomial) Naive Bayes.
    # Here it can be played around with
    # Logistic Regression as well.
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    return None

if __name__ == "__main__":
    main()

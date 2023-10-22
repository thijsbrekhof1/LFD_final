#!/usr/bin/env python

"""
Baseline model for classifying offensive language
Students: Joris Ruitenbeek (s4940148), Thijs Brekhof (s3746135), Niels Top (s4507754)
"""

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def create_arg_parser():
    """Function that builds up all the arguments used in this script.
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf",
        "--train_file",
        default="train.tsv",
        type=str,
        help="Train file to learn from (default train.txt)",
    )
    parser.add_argument(
        "-df",
        "--dev_file",
        default="dev.tsv",
        type=str,
        help="Dev file to evaluate on (default dev.txt)",
    )

    parser.add_argument(
        "-vec",
        "--vectorizer",
        default="tfidf",
        type=str,
        help="Vectorizer to use (tfidf or count) (default tfidf)",
    )
    args = parser.parse_args()

    return args


def read_corpus(corpus_file):
    """
    Reads the corpus file and gets the documents with labels.
    :param str corpus_file: Path to the corpus file.
    :return: the document
    :return: the labels
    """
    documents = []
    labels = []

    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            labels.append(line.split()[-1])
            documents.append(line.split()[:-1])

    return documents, labels


def identity(inp):
    """Dummy function that just returns the input"""
    return inp


def get_vectorizer(vectorizer_name):
    """
    This function returns the vectorizer to use with the correct parameters. This can be tfidf or count.

    :param str vectorizer_name: The vectorizer to use, tfidf or count.
    :return: Returns the chosen vectorizer with the correct parameters set.
    """

    if vectorizer_name == "tfidf":
        return TfidfVectorizer(
            preprocessor=identity,
            tokenizer=identity

        )
    elif vectorizer_name == "count":
        return CountVectorizer(
            preprocessor=identity,
            tokenizer=identity
        )
    else:
        print("Invalid vectorizer name. Choose from 'tfidf' or 'count'.")
        exit(-1)


if __name__ == "__main__":
    # Parse command line arguments
    args = create_arg_parser()

    # Load train and test data with their labels
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Create vectorizer
    vec_word = get_vectorizer(
        args.vectorizer)

    # Create classifier pipeline
    cls = svm.SVC(random_state=1234)

    classifier = Pipeline([("vec", vec_word), ("cls", cls)])

    # fit the classifier on the training data
    classifier.fit(X_train, Y_train)

    # Make predictions
    Y_pred = classifier.predict(X_test)

    # Print the algorithm being used, classification report, and confusion matrix
    print(f"The algorithm being used: {cls}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))

    cf_matrix = confusion_matrix(Y_test, Y_pred)
    index = ["NOT", "OFF"]
    columns = ["NOT", "OFF"]
    cm_df = pd.DataFrame(cf_matrix, columns, index)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.show()

#!/usr/bin/env python

"""
Baseline model for classifying offensive language
Students: Joris Ruitenbeek (s4940148), Thijs Brekhof (s3746135), Niels Top (s4507754)
"""

import argparse
import sys

import spacy

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

allowed_algorithms = {
    "nb": MultinomialNB(),
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "kn": KNeighborsClassifier(),
    "svm1": svm.SVC(),
    "svm2": svm.LinearSVC(),
}

def create_arg_parser():
    """Function that builds up all the arguments used in this script.
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf",
        "--train_file",
        default="train.txt",
        type=str,
        help="Train file to learn from (default train.txt)",
    )
    parser.add_argument(
        "-df",
        "--dev_file",
        default="dev.txt",
        type=str,
        help="Dev file to evaluate on (default dev.txt)",
    )
    parser.add_argument(
        "-testf",
        "--test_file",
        default="test.txt",
        type=str,
        help="Dev file to evaluate on (default test.txt)",
    )

    parser.add_argument(
        "-vec",
        "--vectorizer",
        default="tfidf",
        type=str,
        help="Vectorizer to use (tfidf or count) (default tfidf)",
    )

    return args


def read_corpus(corpus_file, use_sentiment):

    """
    Reads the corpus file and gets the documents with labels.
    :param str corpus_file: Path to the corpus file.
    :param bool use_sentiment: Switch between the sentiment (pos, neg) or the categorical
    :return: the document
    :return: the labels
    """
    documents = []
    labels = []

    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels


def identity(inp):
    """Dummy function that just returns the input"""
    return inp

def get_vectorizer(vectorizer_name):
    """
    This function returns the vectorizer to use with the correct parameters. This can be tfidf or count.

    :param str vectorizer_name: The vectorizer to use, tfidf or count.
    :param num ngram_range: The ngram range to use.
    :param num max_df: The max df to use.
    :param num min_df: The min df to use.
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


def get_algorithm(algorithm_name):
    """
    Checks if the requested algorithm is available and returns the sklearn function for it.
    If the hyperparameter argument is selected, this function also enables the optimized hyperparameters.

    :param str algorithm_name: Name of the algorithm
    :return: The used sklearn function of the algorithm
    """

    if algorithm_name not in allowed_algorithms:
        print(f"Please use a valid algorithm ({', '.join(allowed_algorithms.keys())})")
        exit(-1)

    if args.hyperparameter:
        if args.algorithm == "nb":
            return MultinomialNB(alpha=0.5)
        elif args.algorithm == "dt":
            return DecisionTreeClassifier(
                criterion="gini", max_depth=20, min_samples_leaf=5, random_state=5
            )
        elif args.algorithm == "rf":
            return RandomForestClassifier(
                criterion="gini", max_depth=20, min_samples_leaf=2, random_state=5
            )
        elif args.algorithm == "kn":
            return KNeighborsClassifier(
                n_neighbors=50, weights="distance", p=2, leaf_size=10
            )
        elif args.algorithm == "svm1":
            return svm.SVC(C=0.9, gamma="scale", kernel="linear")
        elif args.algorithm == "svm2":
            return svm.LinearSVC(C=1, loss='hinge')
    else:
        return allowed_algorithms[algorithm_name]

if __name__ == "__main__":
    # Parse command line arguments
    args = create_arg_parser()

    # Load train and test data with their labels
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert lists of tokens into strings for each document
    X_train_strings = [" ".join(tokens) for tokens in X_train]
    X_test_strings = [" ".join(tokens) for tokens in X_test]

    # Create vectorizer

    vec_word = get_vectorizer(
        args.vectorizer, args.ngram_range, args.max_df, args.min_df)

    # Create classifier pipeline
    cls = get_algorithm(args.algorithm)
    classifier = Pipeline([("vec", vec_word), ("cls", cls)])




    X_train_combined = X_train
    X_test_combined = X_test


    # fit the classifier on the training data
    classifier.fit(X_train_combined, Y_train)

    # Make predictions
    Y_pred = classifier.predict(X_test_combined)

    # Print the algorithm being used, classification report, and confusion matrix
    print(f"The algorithm being used: {cls}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))
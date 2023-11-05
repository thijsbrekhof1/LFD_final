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
import emoji
from wordsegment import load, segment

allowed_algorithms = {
    "svm1": svm.SVC(verbose=1),
    "svm2": svm.LinearSVC(verbose=1)
}


def create_arg_parser():
    """Function that builds up all the arguments used in this script.
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-alg",
        "--algorithm",
        default="svm",
        type=str,
        help=f"Algorithm to use (options {', '.join(allowed_algorithms.keys())}) (default svm)",
    )
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
        help="Vectorizer to use (tfidf or count) (default tfidf), we use tfidf as our baseline vectorizer",
    )
    # Emoji to textual representation
    parser.add_argument(
        "-dem",
        "--demojize",
        action="store_true",
        help="Demojize the input to rewrite emoji's to their textual representation e.g.,  ❤ -> :heart: ",

    )
    # Emoji to natural language
    parser.add_argument(
        "-demclean",
        "--demojize_clean",
        action="store_true",
        help="Demojize the input to rewrite emoji's to natural language in order to preserve semantic meaning eg., "
             "❤ -> heart",

    )

    parser.add_argument(
        "-seg",
        "--wordsegment",
        action="store_true",
        help="Perform wordsegmentation on hashtags to better detect profanity and other offensive language ",

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
            # emoji substitution for textual representation
            if args.demojize:
                line = emoji.demojize(line)
            # emoji substitution for natural language
            elif args.demojize_clean:
                line = emoji.demojize(line)
                for word in line.split():
                    if word[0] == ":" and word[-1] == ":":
                        line = line.replace(word, " ".join(segment(word)))
            # Hashtag segmentation
            if args.wordsegment:
                for word in line.split():
                    if "#" in word:
                        line = line.replace(word, " ".join(segment(word)))

            documents.append(line.split()[:-1])
            labels.append(line.split()[-1])

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


def get_algorithm(algorithm_name):
    """
    Checks if the requested algorithm is available and returns the sklearn function for it.

    :param str algorithm_name: Name of the algorithm
    :return: The used sklearn function of the algorithm
    """

    if algorithm_name not in allowed_algorithms:
        print(f"Please use a valid algorithm ({', '.join(allowed_algorithms.keys())})")
        exit(-1)

    else:
        return allowed_algorithms[algorithm_name]


if __name__ == "__main__":
    # Parse command line arguments
    args = create_arg_parser()

    # Loading the word segmentation alg when needed
    if args.wordsegment or args.demojize_clean:
        load()
    # Load train and test data with their labels
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Create vectorizer
    vec_word = get_vectorizer(
        args.vectorizer)

    # Create classifier pipeline
    cls = get_algorithm(args.algorithm)

    classifier = Pipeline([("vec", vec_word), ("cls", cls)])

    # fit the classifier on the training data
    classifier.fit(X_train, Y_train)

    # Make predictions
    Y_pred = classifier.predict(X_test)

    # Print the algorithm being used, classification report, and confusion matrix
    print(f"The algorithm being used: {cls}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, digits=3))

    cf_matrix = confusion_matrix(Y_test, Y_pred)
    index = ["NOT", "OFF"]
    columns = ["NOT", "OFF"]
    cm_df = pd.DataFrame(cf_matrix, columns, index)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.show()

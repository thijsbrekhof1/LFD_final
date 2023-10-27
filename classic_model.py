#!/usr/bin/env python

"""
Classic model for classifying offensive language
Students: Joris Ruitenbeek (s4940148), Thijs Brekhof (s3746135), Niels Top (s4507754)
"""

import argparse
import sys

import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import emoji
from wordsegment import load, segment

allowed_algorithms = {
    "svm1": svm.SVC(verbose=1),
    "svm2": svm.LinearSVC(verbose=1)
}

# This is where the spacy model is initialized which is used for lemmatization and obtaining POS tags
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print('SpaCy model "en_core_web_sm" is required to run this program with lemmatization and/or POS tags.\n'
          'Please run "python3 -m spacy download en_core_web_sm" and try again.',
          file=sys.stderr)


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
        help="Train file to learn from (default train.tsv)",
    )
    parser.add_argument(
        "-df",
        "--dev_file",
        default="dev.tsv",
        type=str,
        help="Dev file to evaluate on (default dev.tsv)",
    )

    parser.add_argument(
        "-gr",
        "--grid",
        action="store_true",
        help="Use grid_search to determine optimal hyperparameters",
    )
    parser.add_argument(
        "-hy",
        "--hyperparameter",
        action="store_true",
        help="Use the optimal hyperparameters determined prior",
    )
    parser.add_argument(
        "-vec",
        "--vectorizer",
        default="tfidf",
        type=str,
        help="Vectorizer to use (tfidf or count) (default tfidf)",
    )
    parser.add_argument(
        "-ngram",
        "--ngram_range",
        default="1,1",
        type=str,
        help="N-gram range (e.g., '1,1' for unigrams, '1,2' for unigrams and bigrams) (default 1,1)",
    )
    parser.add_argument(
        "-maxdf",
        "--max_df",
        default=1.0,
        type=float,
        help="Maximum document frequency for feature selection (default 1)",
    )
    parser.add_argument(
        "-mindf",
        "--min_df",
        default=0.0,
        type=float,
        help="Minimum document frequency for feature selection (default 1)",
    )
    parser.add_argument(
        "-lem",
        "--lemmatize",
        action="store_true",
        help="Lemmatize input words using NLTK",
    )
    parser.add_argument(
        "-tag",
        "--tagger",
        action="store_true",
        help="Use POS tagging as additional input features",
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
            if args.demojize:
                line = emoji.demojize(line)

            elif args.demojize_clean:
                line = emoji.demojize(line)
                for word in line.split():
                    if word[0] == ":" and word[-1] == ":":
                        line = line.replace(word, " ".join(segment(word)))

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


def lemmatize_text(text):
    """
    Lemmatizes input text with spacy.

    :param str text: Input text
    :return: A string of lemmatized text
    """
    # doc = nlp(' '.join(text))
    lemmatized = [token.lemma_ for token in nlp(text)]

    return " ".join(lemmatized)


def spacy_pos(text):
    """
    This function returns the part of speech (POS) tags for each word in a given input.

    :param str text: Input text
    :return: Returns the POS tags in a list
    """

    # Tokenize the text using spaCy
    doc = nlp(text)

    # Extract the POS tags and join them into a string
    pos_tags = " ".join([token.pos_ for token in doc])

    return pos_tags


def get_vectorizer(vectorizer_name, ngram_range, max_df, min_df):
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
            tokenizer=identity,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
        )
    elif vectorizer_name == "count":
        return CountVectorizer(
            preprocessor=identity,
            tokenizer=identity,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
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
        if args.algorithm == "svm1":
            return svm.SVC(C=1.0, gamma="scale", kernel="linear")
        elif args.algorithm == "svm2":
            return svm.LinearSVC(C=1.0, loss='squared_hinge')
    else:
        return allowed_algorithms[algorithm_name]


def tune_param():
    """
    Performs gridsearch to fine tune the hyperparameters of the selected type of algorithm
    to determine optimal performance.
    """
    # for naive bayes
    if args.algorithm == "svm1":
        params = {
            "cls__C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
            "cls__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "cls__gamma": ["scale", "auto", 1, 0.1, 0.01],
        }
    # For Linear Support Vector Classification.
    elif args.algorithm == "svm2":
        params = {
            "cls__loss": ["hinge", "squared_hinge"],
            "cls__C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
        }
    else:
        params = {}
    # Perform grid search, verbose entails how detailed the output should be and n_jobs specifies the number of cores
    # your processor should use simultaneously to get results faster, change to -1 to use all cores.
    grid_search = GridSearchCV(classifier, params, verbose=2, n_jobs=6)
    grid_search.fit(X_train, Y_train)

    print("Best Score: ", grid_search.best_score_)
    print("Best Params: ", grid_search.best_params_)


if __name__ == "__main__":
    # Parse command line arguments
    args = create_arg_parser()

    if args.wordsegment or args.demojize_clean:
        load()

    # Load train and test data with their labels
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Convert lists of tokens into strings for each document
    X_train_strings = [" ".join(tokens) for tokens in X_train]
    X_test_strings = [" ".join(tokens) for tokens in X_test]

    # Create vectorizer
    if args.ngram_range:
        ngrams = args.ngram_range.split(",")
        vec_word = get_vectorizer(
            args.vectorizer, (int(ngrams[0]), int(ngrams[1])), args.max_df, args.min_df
        )

    else:
        vec_word = get_vectorizer(
            args.vectorizer, args.ngram_range, args.max_df, args.min_df
        )

    # Create classifier pipeline
    cls = get_algorithm(args.algorithm)
    classifier = Pipeline([("vec", vec_word), ("cls", cls)])

    if args.lemmatize:
        # If lemmatization is enabled, apply it to the data
        X_train_strings = [lemmatize_text(text) for text in X_train_strings]
        X_test_strings = [lemmatize_text(text) for text in X_test_strings]

    if args.tagger:
        # If POS tagging is enabled, create a CountVectorizer for POS tags
        pos = CountVectorizer(tokenizer=spacy_pos)
        vec_pos = pos

        # Combine word vectors and POS vectors using FeatureUnion
        union = FeatureUnion([("word", vec_word), ("pos", vec_pos)])

        # Fit and transform the data
        X_train_combined = union.fit_transform(X_train_strings).toarray()
        X_test_combined = union.transform(X_test_strings).toarray()

    else:
        # If POS tagging is not enabled, only use word vectors
        X_train_combined = X_train
        X_test_combined = X_test

    if args.grid:
        # If grid search for hyperparameter tuning is enabled, perform it
        tune_param()
    else:
        # If not using grid search, fit the classifier on the training data
        classifier.fit(X_train_combined, Y_train)

        # Make predictions
        Y_pred = classifier.predict(X_test_combined)

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

import emoji


def read_corpus(corpus_file):
    """
    Reads the corpus file and gets the documents with labels.
    :param str corpus_file: Path to the corpus file.
    :return: the document

    """
    documents = []

    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            documents.append(line.split()[:-1])


    return documents


def check_emoji(tweets):
    total_emoji = 0
    emoji_tweets = 0

    for tweet in tweets:
        if emoji.emoji_list(tweet):
            emoji_tweets += 1
            total_emoji += len(emoji.emoji_list(tweet))
    return total_emoji, emoji_tweets

def check_hastag(tweets):
    total_hashtag = 0
    hashtag_tweets = 0

    for tweet in tweets:
        if "#" in ' '.join(tweet):
            hashtag_tweets += 1
            for word in tweet:
                if word[0] == "#":
                    total_hashtag += 1
    return total_hashtag, hashtag_tweets


def main():
    X_train = read_corpus("train.tsv")
    X_dev = read_corpus("dev.tsv")
    X_test = read_corpus("test.tsv")

    # Train
    total_emoji_train, emoji_tweets_train = check_emoji(X_train)
    total_hashtag_train, hashtag_tweets_train = check_hastag(X_train)

    print("Total emoji's in train set: {}".format(total_emoji_train))
    print("Total amount of tweets containing emoji's in train set: {}".format(emoji_tweets_train))
    print("Total hashtags in train set: {}".format(total_hashtag_train))
    print("Total amount of tweets containing hashtags in train set: {}".format(hashtag_tweets_train))

    print("\n")
    # Dev
    total_emoji_dev, emoji_tweets_dev = check_emoji(X_dev)
    total_hashtag_dev, hashtag_tweets_dev = check_hastag(X_dev)

    print("Total emoji's in dev set: {}".format(total_emoji_dev))
    print("Total amount of tweets containing emoji's in dev set: {}".format(emoji_tweets_dev))
    print("Total hashtags in dev set: {}".format(total_hashtag_dev))
    print("Total amount of tweets containing hashtags in dev set: {}".format(hashtag_tweets_dev))

    print("\n")
    # test
    total_emoji_test, emoji_tweets_test = check_emoji(X_test)
    total_hashtag_test, hashtag_tweets_test = check_hastag(X_test)
    print("Total emoji's in test set: {}".format(total_emoji_test))
    print("Total amount of tweets containing emoji's in test set: {}".format(emoji_tweets_test))
    print("Total hashtags in test set: {}".format(total_hashtag_test))
    print("Total amount of tweets containing hashtags in test set: {}".format(hashtag_tweets_test))
    print("\n")


main()

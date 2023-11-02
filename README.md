# LFD_final
This is the github repo that we used while working on our final LFD project, regarding offensive language detection

# How to run our code

## General notes
Download our data (train/dev/test .tsv file) and put it in the same folder as the model you want to be using. (Or clone the repository)

Download all packages and modules specified in our import statements.

## Checking statistics about the dataset
Statistics regarding emojis and hashtags can be found when running 'check_stats.py'. No additionaly command-line arguments need to be specified for this. 

## Baseline
Our baseline model can be found under 'baseline.py'. The program does not require any default command-line arguments to be specified to run, though you can optionally change settings like which algorithm to use by specifying command-line arguments. Running the program with the argument -h will show all possible command-line arguments and their options.  

## Classic model(s)
Our baseline model can be found under 'classic_model'. The program does not require any default command-line arguments to be specified to run, though you can optionally change settings like which algorithm to use by specifying command-line arguments. Running the program with the argument -h will show all possible command-line arguments and their options.  

## LSTM
nog ff toevoegen

## PTLM
nog ff toevoegen

# How to install dependencies

# How to preprocess our data
All preprocessing steps that we apply on our data happen within the scripts for each of our models. 

They can be specified using the command-line arguments, namely:

 -dem for substituting emojis for a textual representation;
 -demclean for substituting emojis for natural language;
 -seg for applying hashtag segmentation
 
# How to train the models on the data
Our scripts all apply training to models based on our dataset. 

It is possible to specify the training file and test file by using the -tf and -df command line arguments respectively, followed by the name of the file you want to utilize.

# Authors
Thijs Brekhof    https://github.com/thijsbrekhof1

Niels Top        https://github.com/nitgi

Joris Ruitenbeek https://github.com/iJoris


# Iets om mee te nemen bij het werken aan de LSTM:
The top nonBERT model, MIDAS, is ranked sixth. They used
an ensemble of CNN and BLSTM+BGRU, together with Twitter word2vec embeddings (Godin
et al., 2015) and token/hashtag normalization
(Van de shared task over deze data)

# Miss interessant mbt emojis
NULI was ranked 1st, 4th, and 18th on sub-tasks
A, B, and C, respectively. They experimented
with different models including linear models, LSTM, and pre-trained BERT with finetuning on the OLID dataset. Their final
submissions for all three subtasks only used
BERT, which performed best during development. They also used a number of preprocessing techniques such as hashtag segmentation and emoji substitution.

## from NULI paper https://aclanthology.org/S19-2011/:
Emoji substitution We use one online emoji
project on github (https://github.com/carpedm20/emoji) which could map the emoji unicode to substituted phrase. We treat such phrases
into regular English phrase thus it could maintain their semantic meanings, especially when the dataset size is limited.

## From the same paper:
HashTag segmentation (using https://github.com/grantjenks/python-wordsegment) The HashTag becomes
a popular culture cross multi social networks, including Twitter, Instagram, Facebook etc. In order
to detect whether the HashTag contains profanity words, we apply word segmentation using one
open source on the github 3
. One typical example
would be ‘#LunaticLeft’ is segmented as ‘Lunatic
Left’ which is obviously offensive in this case.
using 

# LFD_final
This is the github repo that we used while working on our final LFD project, regarding offensive language detection

# How to run our code

## General notes
Download our data (train/dev/test .tsv file) and put it in the same folder as the model you want to be using. (Or clone the repository)

Download all packages and modules specified in our import statements.

## Checking statistics about the dataset
Statistics regarding emojis and hashtags can be found when running 'check_stats.py'. No additional command-line arguments need to (nor can) be specified for this. 

## Baseline
Our baseline model can be found under 'baseline.py'. The program does not require any default command-line arguments to be specified to run, though you can optionally change settings like which algorithm to use by specifying command-line arguments. Running the program with the argument -h will show all possible command-line arguments and their options.  

## Classic model(s)
Our baseline model can be found under 'classic_model'. The program does not require any default command-line arguments to be specified to run, though you can optionally change settings like which algorithm to use by specifying command-line arguments. Running the program with the argument -h will show all possible command-line arguments and their options.  

## LSTM
nog ff toevoegen

## PTLM
nog ff toevoegen

# How to preprocess our data
All preprocessing steps that we apply on our data happen within the scripts for each of our models. 

They can be specified using the command-line arguments, namely:

 -dem for substituting emojis for a textual representation;
 
 -demclean for substituting emojis for natural language;
 
 -seg for applying hashtag segmentation
 
# How to train the models on the data
It is possible to specify the training file and test file by using the -tf and -df command line arguments respectively, followed by the name of the file you want to utilize.

e.g., -tf train.tsv -df test.tsv

# Authors
Thijs Brekhof    https://github.com/thijsbrekhof1

Niels Top        https://github.com/nitgi

Joris Ruitenbeek https://github.com/iJoris


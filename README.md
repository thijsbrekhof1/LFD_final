# LFD_final
This is the github repo that we used while working on our final LFD project, regarding offensive language detection

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
https://aclanthology.org/S19-2011/

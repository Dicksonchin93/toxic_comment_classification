# Toxic Comment Classification

This is my codes for the toxic comment classification competition hosted in [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Fully modified to another level from the base code [here](https://github.com/conversationai/unintended-ml-bias-analysis/tree/master/unintended_ml_bias)


To download datasets please run get_data.sh
## The Task
The dataset comprises of comments from Wikipedia’s talk page edits. It is a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

 > *  `toxic`
 > *  `severe_toxic`
 > *  `obscene`
 > *  `threat`
 > *  `insult`
 > *  `identity_hate`
 
 
## The Approach

Creating an ensemble model which predicts a probability of each type of toxicity for each comment.Full explaination of my approach is documented [here](https://medium.com/@dickson_chin93/my-solution-to-achieve-top-1-in-a-novel-data-science-nlp-competition-db8db2ee356a)



## Install Pre-requisites

run install.sh and then run 
pip install -r requirements.txt

## Tips

- Make sure embeddings original preprocessing is used to ensure highest percentage of embeddings can be imported

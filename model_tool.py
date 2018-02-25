"""Train a Toxicity model using Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle
import json
import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import concatenate
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.layers import Embedding
from keras.layers import Embedding
from keras import regularizers
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import CuDNNGRU
from keras.layers import GRU
from keras.models import load_model
import fastText
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
import nltk


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from gensim.models.keyedvectors import KeyedVectors
#import enchant
#import splitter
import h5py
from sklearn.metrics import log_loss
from sklearn.metrics import auc
import string
import re
from nltk.corpus import stopwords
#from fastText import load_model


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
        


print('HELLO from model_tool')

#DEFAULT_EMBEDDINGS_PATH = 'data/glove.6B/glove.840B.300d.txt'
#DEFAULT_EMBEDDINGS_PATH = 'crawl-300d-2M.vec'
DEFAULT_EMBEDDINGS_PATH = 'wiki.en.vec'
DEFAULT_MODEL_DIR = 'models'

DEFAULT_HPARAMS = {
    'max_sequence_length': 150,
    'max_num_words': 195121,
    'embedding_dim': 300,
    'embedding_trainable': False,
    'learning_rate': 0.001,
    'stop_early': False,
    'es_patience': 1,  # Only relevant if STOP_EARLY = True
    'es_min_delta': 0,  # Only relevant if STOP_EARLY = True
    'batch_size': 128,
    'epochs': 1,
    'dropout_rate': 0.5,
    'cnn_filter_sizes': [128, 128, 128],
    'cnn_kernel_sizes': [5, 5, 5],
    'cnn_pooling_sizes': [5, 5, 40],
    'verbose': True
}


def compute_auc(y_true, y_pred):
  try:
    return metrics.roc_auc_score(y_true, y_pred)
  except ValueError:
    return np.nan


### Model scoring

# Scoring these dataset for dozens of models actually takes non-trivial amounts
# of time, so we save the results as a CSV. The resulting CSV includes all the
# columns of the original dataset, and in addition has columns for each model,
# containing the model's scores.
def score_dataset(df, models, text_col):
    """Scores the dataset with each model and adds the scores as new columns."""
    for model in models:
        name = model.get_model_name()
        print('{} Scoring with {}...'.format(datetime.datetime.now(), name))
        df[name] = model.predict(df[text_col])

def load_maybe_score(models, orig_path, scored_path, postprocess_fn):
    if os.path.exists(scored_path):
        print('Using previously scored data:', scored_path)
        return pd.read_csv(scored_path)

    dataset = pd.read_csv(orig_path)
    postprocess_fn(dataset)
    score_dataset(dataset, models, 'text')
    print('Saving scores to:', scored_path)
    dataset.to_csv(scored_path)
    return dataset

def postprocess_madlibs(madlibs):
    """Modifies madlibs data to have standard 'text' and 'label' columns."""
    # Native madlibs data uses 'Label' column with values 'BAD' and 'NOT_BAD'.
    # Replace with a bool.
    madlibs['label'] = madlibs['Label'] == 'BAD'
    madlibs.drop('Label', axis=1, inplace=True)
    madlibs.rename(columns={'Text': 'text'}, inplace=True)

def postprocess_wiki_dataset(wiki_data):
    """Modifies Wikipedia dataset to have 'text' and 'label' columns."""
    wiki_data.rename(columns={'is_toxic': 'label',
                              'comment': 'text'},
                     inplace=True)



class ToxModel():
  """Toxicity model."""

  def __init__(self,
               model_name=None,
               model_dir=DEFAULT_MODEL_DIR,
               embeddings_path=DEFAULT_EMBEDDINGS_PATH,
               hparams=None):
    self.model_dir = model_dir
    self.embeddings_path = embeddings_path
    self.model_name = model_name
    self.model = None
    self.tokenizer = None
    self.hparams = DEFAULT_HPARAMS.copy()
    if hparams:
      self.update_hparams(hparams)
    if model_name:
      self.load_model_from_name(model_name)
    self.print_hparams()

  def print_hparams(self):
    print('Hyperparameters')
    print('---------------')
    for k, v in self.hparams.iteritems():
      print('{}: {}'.format(k, v))
    print('')

  def update_hparams(self, new_hparams):
    self.hparams.update(new_hparams)

  def get_model_name(self):
    return self.model_name

  def save_hparams(self, model_name):
    self.hparams['model_name'] = model_name
    with open(
        os.path.join(self.model_dir, '%s_hparams.json' % self.model_name),
        'w') as f:
      json.dump(self.hparams, f, sort_keys=True)

  def load_model_from_name(self, model_name):
    self.model = load_model(
        os.path.join(self.model_dir, '%s_model.h5' % model_name))
    self.tokenizer = cPickle.load(
        open(
            os.path.join(self.model_dir, '%s_tokenizer.pkl' % model_name),
            'rb'))
    with open(
        os.path.join(self.model_dir, '%s_hparams.json' % self.model_name),
        'r') as f:
      self.hparams = json.load(f)
  
  def load_pretrained_model(self):
    weights_path = self.model_name + '_model.h5'
    f = h5py.File(weights_path)
    layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
    for i in layer_dict.keys():
      weight_names = f["model_weights"][i].attrs["weight_names"]
      weights = [f["model_weights"][i][j] for j in weight_names]
      index = layer_names.index(i)
      self.model.layers[index].set_weights(weights)
    for layer in self.model.layers[:12]:
      layer.trainable = False


  def fit_and_save_tokenizer(self, texts):
    """Fits tokenizer on texts and pickles the tokenizer state."""
    self.tokenizer = Tokenizer(num_words=self.hparams['max_num_words'])
    self.tokenizer.fit_on_texts(texts)
    cPickle.dump(self.tokenizer,
                 open(
                     os.path.join(self.model_dir,
                                  '%s_tokenizer.pkl' % self.model_name ), 'wb'))

  def prep_text(self, texts):
    """Turns text into into padded sequences.

    The tokenizer must be initialized before calling this method.

    Args:
      texts: Sequence of text strings.

    Returns:
      A tokenized and padded text sequence as a model input.
    """
    text_sequences = self.tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        text_sequences, maxlen=self.hparams['max_sequence_length'])
        
  def Sanitize(self, df):
    repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
    }

    keys = [i for i in repl.keys()]
    ltr = df["comment_text"].tolist()
    new_train_data = []
    for i in ltr:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + " "
        new_train_data.append(xx)
    df["comment_text"] = new_train_data
    comment = 'comment' if 'comment' in df else 'comment_text'
    stopwords_en = stopwords.words('english')
    df[comment] = df[comment].str.lower().str.replace('newline_token', ' ')
    df[comment] = df[comment].fillna('erikov')
    #special symbols
    df[comment] = df[comment].apply(lambda x : " ".join(re.findall('[\w]+',x)))
    #stop words
    df[comment] = df[comment].apply(lambda x: self.remove_stopwords(x, stopwords_en))
    #websites
    df[comment] = df[comment].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df[comment] = df[comment].replace(r'@[^\s]+[\s]?', '', regex=True)
    #ip address
    df[comment] = df[comment].replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', regex=True)
    
    df[comment] = df[comment].replace('&', ' and ', regex=True)
    df[comment] = df[comment].replace('@', ' at ', regex=True)
    df[comment] = df[comment].replace('0', ' zero ', regex=True)
    df[comment] = df[comment].replace('1', ' one ', regex=True)
    df[comment] = df[comment].replace('2', ' two ', regex=True)
    df[comment] = df[comment].replace('3', ' three ', regex=True)
    df[comment] = df[comment].replace('4', ' four ', regex=True)
    df[comment] = df[comment].replace('5', ' five ', regex=True)
    df[comment] = df[comment].replace('6', ' six ', regex=True)
    df[comment] = df[comment].replace('7', ' seven ', regex=True)
    df[comment] = df[comment].replace('8', ' eight ', regex=True)
    df[comment] = df[comment].replace('9', ' nine ', regex=True)
    #multi space
    df[comment] = df[comment].replace('\s+', ' ', regex=True)
    
    return df

    
  def remove_stopwords(self, s, stop_words):
    '''For removing stop words
    '''
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s

  def stem(self, text, stemmer=nltk.PorterStemmer()):
    def stem_and_join(row):
        row["comment_text"] = list(map(lambda str: stemmer.stem(str.lower()), row["comment_text"]))
        return row

    text = text.apply(stem_and_join, axis=1)
    return text
  
  def char_text(self, texts):
    """Turns text into char encoded matrix
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    examples = []
    i = 0
    for sentence in texts:
        text_end_extracted = self.extract_end(list(sentence.lower()))  #To make sure no text go pass the calculated max frame length
        padded = self.pad_sentence(text_end_extracted)# To pad to fixed frame length of 1014
        text_int8_repr = self.string_to_int8_conversion(padded, alphabet) #change to integer
        examples.append(text_int8_repr)
        i += 1
        if i % 10000 == 0:
            print("Non-neutral instances processed: " + str(i))
            
    return examples
  
  def load_data(self, texts):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    #alphabet = "abcdefghijklmnopqrstuvwxyz"
    examples = self.char_text(texts)
    x = np.array(examples, dtype=np.int8)

    print("x_char_seq_ind=" + str(x.shape))

    return x
    
  def extract_end(self, char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq
  
  def pad_sentence(self, char_seq, padding_char=" "):
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq
  
  def string_to_int8_conversion(self, char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x

  def load_embeddings(self):
    """Loads word embeddings."""
    word_vectors = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)

    embeddings_index = {}
    with open(self.embeddings_path) as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1,
                                      self.hparams['embedding_dim']))
    num_words_in_embedding = 0
    num_words_replaced = 0
    num_words_fail_replaced = 0
    for word, i in self.tokenizer.word_index.items():
      if (i >= (len(self.tokenizer.word_index) + 1)):
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        num_words_in_embedding += 1
        # words not found in embedding index will be all-zeros.
        self.embedding_matrix[i] = embedding_vector
      else:
        word = ''.join(e for e in word if e.isalnum())
        word = ''.join([f for f in word if not f.isdigit()])
        try:               
          replace_word = word_vectors.most_similar(word)[0][0]
          embedding_vector_second = embeddings_index.get(replace_word)
          if embedding_vector_second is not None:
            num_words_in_embedding += 1
            self.embedding_matrix[i] = embedding_vector_second
            num_words_replaced += 1
          else:
            split_trial = splitter.split(replace_word)
            if split_trial is not "":
              for wrd in split_trial:
                embedding_vector_third = embeddings_index.get(wrd)
                if embedding_vector_third is not None:
                  num_words_in_embedding += 1
                  self.embedding_matrix[i] = embedding_vector_third
                  num_words_replaced += 1                              
                else:
                  num_words_fail_replaced += 1
                  print(" not in vocabulary")
            else:
              d = enchant.Dict("en_US")
              wrd = d.suggest(replace_word)
              if not wrd:
                num_words_fail_replaced += 1
                print("{} not in vocabulary List is empty".format(wrd))
              else:
                wrd = wrd[0]
                wrd = ''.join(e for e in wrd if e.isalnum())
                wrd = ''.join([f for f in wrd if not f.isdigit()])
                embedding_vector_fourth = embeddings_index.get(wrd)
                if embedding_vector_fourth is not None:
                  num_words_in_embedding += 1
                  self.embedding_matrix[i] = embedding_vector_fourth
                  num_words_replaced += 1                              
                else:
                  num_words_fail_replaced += 1
                  print(" not in vocabulary")

        except KeyError:
          d = enchant.Dict("en_US")
          if word == "":
            num_words_fail_replaced += 1
            print("{} not in vocabulary".format(word))
          else:
            wrd = d.suggest(word)
            if not wrd:
              num_words_fail_replaced += 1
              print("{} not in vocabulary List is empty".format(wrd))
            else:
              wrd = wrd[0]
              wrd = ''.join(e for e in wrd if e.isalnum())
              wrd = ''.join([f for f in wrd if not f.isdigit()])
              embedding_vector_fourth = embeddings_index.get(wrd)
              if embedding_vector_fourth is not None:
                num_words_in_embedding += 1
                self.embedding_matrix[i] = embedding_vector_fourth
                num_words_replaced += 1                              
              else:
                split_trial = splitter.split(wrd)                
                if split_trial is not "":
                  for wrds in split_trial:
                    embedding_vector_third = embeddings_index.get(wrds)
                    if embedding_vector_third is not None:
                      num_words_in_embedding += 1
                      self.embedding_matrix[i] = embedding_vector_third
                      num_words_replaced += 1                              
                    else:
                      num_words_fail_replaced += 1
                      print("{} not in vocabulary".format(wrds))
                else:
                  num_words_fail_replaced += 1
                  print("{} not in vocabulary".format(wrd))
      
    print("Number of words replaced: {}  Failed: {}".format(num_words_replaced, num_words_fail_replaced))
    
  def load_embeddings_fast(self):
    """Loads word embeddings."""
    #word_vectors = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)

    embeddings_index = {}
    with open(self.embeddings_path) as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1,
                                      self.hparams['embedding_dim']))
    num_words_in_embedding = 0
    num_words_replaced = 0
    num_words_fail_replaced = 0
    for word, i in self.tokenizer.word_index.items():
      if (i >= (len(self.tokenizer.word_index) + 1)):
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        num_words_in_embedding += 1
        # words not found in embedding index will be all-zeros.
        self.embedding_matrix[i] = embedding_vector
  
  def load_embeddings_fast_bin(self):
    """Loads word embeddings."""
    #word_vectors = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)
    '''
    embeddings_index = {}
    with open(self.embeddings_path) as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    '''
    ft_model = fastText.load_model('wiki.en.bin')
    n_features = ft_model.get_dimension()
    
    self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1,
                                      self.hparams['embedding_dim']))
    num_words_in_embedding = 0
    num_words_replaced = 0
    num_words_fail_replaced = 0
    for word, i in self.tokenizer.word_index.items():
      if (i >= (len(self.tokenizer.word_index) + 1)):
        continue
      embedding_vector = ft_model.get_word_vector(word).astype('float32')
      if embedding_vector is not None:
        num_words_in_embedding += 1
        # words not found in embedding index will be all-zeros.
        self.embedding_matrix[i] = embedding_vector
      
        
  def load_model_FT(self):
    ft_model = load_model('wiki.en.bin')
    n_features = ft_model.get_dimension()
    return ft_model, n_features
    
  def text_to_vector(self, text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    window_length = 250
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x
    
  def df_to_data(self, df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    window_length =250
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x

  def train(self, cl, training_data_path, text_column,
            toxic, severe_toxic, obscene, threat, insult, identity_hate , model_name, model_list,pretrain = False):
    """Trains the model."""
    folds = 10
    self.model_name = model_name
    self.save_hparams(model_name)

    train_data = pd.read_csv(training_data_path)
    train_data['comment_text'] = train_data['comment_text'].fillna('_empty_')
   # valid_data = pd.read_csv(validation_data_path)
   
    print('Fitting tokenizer...')
    self.fit_and_save_tokenizer(train_data[text_column])
    print('Tokenizer fitted!')

    print('Preparing data...')
    #train_data['new'] = train_data.apply(lambda x: [x[toxic], x[severe_toxic], x[obscene], x[threat], x[insult], x[identity_hate]], axis=1)
    #train_data['new'] = train_data['new'].apply(lambda x: np.array(x))
    if (cl == 1):
        tl = train_data[['obscene', 'threat', 'insult', 'identity_hate']].values
    elif (cl == 0):
        tl = train_data[['toxic','severe_toxic']].values
    elif (cl == 2):
        tl = train_data[['toxic','severe_toxic','obscene', 'threat', 'insult', 'identity_hate']].values
    elif (cl == 6):
        tl = train_data[['toxic','severe_toxic','obscene', 'threat', 'insult', 'identity_hate']].values
    #train_text, train_labels = (self.prep_text(train_data[text_column]),
    #                            to_categorical(train_data[toxic]))
    train_text_temp, train_labels_temp = (self.prep_text(train_data[text_column]),
                                tl)
    #CHAR-level                           
    #train_text_temp, train_labels_temp = (self.load_data(train_data[text_column]),
    #                            tl)
    '''
    for id, t in enumerate(train_labels_temp):
        if id < 20:
            print(t)
            print(t.shape)
    '''
    print("train_text_temp shape {} and train_labels_temp shape {}".format(train_text_temp.shape, train_labels_temp.shape))
    #valid_text, valid_labels = (self.prep_text(valid_data[text_column]),
    #                            to_categorical(valid_data[toxic]))
                                
    '''                           
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=10) 
    for train_index, test_index in sss.split(train_text_temp, train_labels_temp):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_text, valid_text = x[train_index], x[test_index]
        train_labels, valid_labels = y[train_index], y[test_index]
     '''   
    print(" ---- ")
    if (cl == 2):
        train_text = train_text_temp
        train_labels = train_labels_temp
    else:
    
        train_text, valid_text, train_labels, valid_labels = train_test_split(train_text_temp, train_labels_temp, test_size = 0.1, random_state = 0)
    
    #print("valid_text shape {} and valid_labels shape {}".format(valid_text.shape, valid_labels.shape))
    print("train_text shape {} and train_labels shape {}".format(train_text.shape, train_labels.shape))
    print('Data prepared!')

    print('Loading embeddings...')
    if (cl != 2):
        self.load_embeddings()
    #self.load_embeddings_fast()
    self.load_embeddings_fast_bin()
    print('Embeddings loaded!')

    print('Building model graph...')
    #self.build_model()
    #self.build_model_two()
    list_models = []
    if (cl == 1):
        self.build_model_three()
        if pretrain:
            self.load_pretrained_model()      
    elif (cl == 0):
        self.build_model_four()
        if pretrain:
            self.load_pretrained_model()
    elif (cl == 2):
    #####FIRST double GRU 
        '''
        self.build_model_five()
        if pretrain:
            self.load_pretrained_model()
        '''

        for fold in range(0, folds):
            #if (fold >= 8):
            model = self.build_model_eight()
            #model_path = os.path.join(self.model_dir, "model{0}_weights.npy".format(fold))
            #weights = np.load(model_path)
            #model.set_weights(weights)
            list_models.append(model)
            #else:
            #list_models.append(model_list[fold])
    elif (cl==6):
        for fold in range(0, folds):
            model = self.build_model_seven()
            list_models.append(model)
        
    #self.model.summary()

    print('Training model...')

    

    
    if (cl == 2):
        models, total_meta = self.train_folds(train_text, train_labels, folds, list_models)
        #models = self.train_leftover_folds(train_text, train_labels, folds, list_models, 8)
        
        print('Model trained!')
        print("Predicting results...")
        random_test = pd.read_csv('cleaned_test_clean.csv')
        #random_test = self.Sanitize(random_test)
        #random_test.to_csv('cleaned_test_clean.csv', index=False)
        X_test = random_test['comment_text'].fillna('_empty_')
        X_test = self.prep_text(X_test)
        #X_test = self.load_data(X_test)
        test_predicts_list = []
        for fold_id, model in enumerate(models):
            model_path = os.path.join(self.model_dir, "model{0}_weights.npy".format(fold_id))
            np.save(model_path, model.get_weights())
        
            test_predicts_path = os.path.join(self.model_dir, "test_predicts{0}.npy".format(fold_id))
            test_predicts = model.predict(X_test, batch_size=self.hparams['batch_size'])
            test_predicts_list.append(test_predicts)
            np.save(test_predicts_path, test_predicts)

        test_predicts = np.ones(test_predicts_list[0].shape)
        for fold_predict in test_predicts_list:
            test_predicts *= fold_predict

        test_predicts **= (1. / len(test_predicts_list))
        test_ids = random_test["id"].values
        test_ids = test_ids.reshape((len(test_ids), 1))
        CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
        test_predicts["id"] = test_ids
        test_predicts = test_predicts[["id"] + CLASSES]
        test_predicts.to_csv('augmentori_pred_fasttext_gru_cv_concat.csv', index=False)
        print('predicted !')
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        subm = pd.read_csv('cleaned_final_train_clean.csv')
        submid = pd.DataFrame({'id': subm["id"]})
        total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns = label_cols)], axis=1)
        total_meta_data.to_csv('augmentori_pred_fasttext_gru_cv_concat_pretrain_meta.csv', index=False)
        print('Meta predicted !')
    else:

        save_path = os.path.join(self.model_dir, '%s_model.h5' % self.model_name )
        callbacks = [
            ModelCheckpoint(
                save_path, save_best_only=True, verbose=self.hparams['verbose'])
        ]


        if self.hparams['stop_early']:
          callbacks.append(
              EarlyStopping(
                  min_delta=self.hparams['es_min_delta'],
                  monitor='val_loss',
                  patience=self.hparams['es_patience'],
                  verbose=self.hparams['verbose'],
                  mode='auto'))

        self.model.fit(
            train_text,
            train_labels,
            batch_size=self.hparams['batch_size'],
            epochs=self.hparams['epochs'],
            validation_data=(valid_text, valid_labels),
            callbacks=callbacks,
            verbose=2)


        print('Model trained!')
        print('Best model saved to {}'.format(save_path))
        print('Loading best model from checkpoint...')
        self.model = load_model(save_path)
        print('Model loaded!')
    

  def build_model(self):
    l2_weight_decay=0.0001
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    for filter_size, kernel_size, pool_size in zip(
        self.hparams['cnn_filter_sizes'], self.hparams['cnn_kernel_sizes'],
        self.hparams['cnn_pooling_sizes']):
      x = self.build_conv_layer(x, filter_size, kernel_size, pool_size)

    x = Flatten()(x)
    x = Dropout(self.hparams['dropout_rate'])(x)
    # TODO(nthain): Parametrize the number and size of fully connected layers
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    #tf.nn.sigmoid_cross_entropy_with_logits
    preds = Dense(6, activation='sigmoid')(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    adam = Adam(lr=self.hparams['learning_rate'])
    self.model = Model(sequence_input, preds)
    self.model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
  
  def build_model_two(self):
    l2_weight_decay=0.0001
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    
    x = self.build_lstm_layer(x)

    #x = Flatten()(x)
    x = Dropout(self.hparams['dropout_rate'])(x)
    # TODO(nthain): Parametrize the number and size of fully connected layers
    x = Dense(50, activation='relu')(x)
    #tf.nn.sigmoid_cross_entropy_with_logits
    x = Dropout(self.hparams['dropout_rate'])(x)
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    preds = Dense(6, activation='sigmoid')(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    adam = Adam(lr=self.hparams['learning_rate'])
    self.model = Model(sequence_input, preds)
    self.model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
  
  def build_model_three(self):
    l2_weight_decay=0.0001
    """Builds model graph."""

    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    modellstm_two = self.get_lstm(x, l2_weight_decay)
    modellstm = self.get_lstm(x, l2_weight_decay)
    
    x = concatenate([modellstm, modellstm_two], axis=1)
    x = Dropout(self.hparams['dropout_rate'])(x)
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    preds = Dense(4, activation='sigmoid')(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    adam = Adam(lr=self.hparams['learning_rate'])
    self.model = Model(sequence_input, preds)
    self.model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
  
  def build_model_four(self):
    l2_weight_decay=0.0001
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    modelcnn = self.get_cnn(x, l2_weight_decay)
    modellstm = self.get_lstm(x, l2_weight_decay)
    
    x = concatenate([modellstm, modelcnn], axis=1)
    x = Dropout(self.hparams['dropout_rate'])(x)
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    preds = Dense(2, activation='sigmoid')(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    adam = Adam(lr=self.hparams['learning_rate'])
    self.model = Model(sequence_input, preds)
    self.model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
        
  def build_model_five(self):
    l2_weight_decay=0.0001
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    
    x = self.get_gru(x, l2_weight_decay)

  
    preds = Dense(6, activation='sigmoid')(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    adam = Adam(lr=self.hparams['learning_rate'])
    #self.model = Model(sequence_input, preds)
    #self.model.compile(
    #    loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    model = Model(sequence_input, preds)
    model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    return model
    
  def build_model_six(self, train_text):
    l2_weight_decay=0.0001
    lstm_dim = 100
    recurrent_units=64
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    model = Sequential()

    model.add(LSTM(lstm_dim,input_shape=(1014, 1), return_sequences=True))
    model.add(Dropout(self.hparams['dropout_rate']))
    model.add(Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True)))
    model.add(Dense(32, activation='relu'))
    model.add(Bidirectional(LSTM(lstm_dim, return_sequences=False)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    model.add(Dense(6, activation='sigmoid'))

    adam = Adam(lr=self.hparams['learning_rate'])
    model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    return model
  
  def build_model_seven(self):
    l2_weight_decay=0.0001
    lstm_dim = 100
    recurrent_units=64
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    model = Sequential()

    model.add(Conv1D(256, 7, activation='linear', input_shape=(1014, 1)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Conv1D(256, 3, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.5))

    model.add(Dense(150, activation='linear', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(6, activation='sigmoid'))

    adam = Adam(lr=self.hparams['learning_rate'])
    model.compile(
        loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    return model
  
  def build_model_eight(self):
    l2_weight_decay=0.0001
    """Builds model graph."""
    sequence_input = Input(
        shape=(self.hparams['max_sequence_length'],), dtype='int32')
    embedding_layer = Embedding(
        self.hparams['max_num_words'],
        self.hparams['embedding_dim'],
        weights=[self.embedding_matrix],
        input_length=self.hparams['max_sequence_length'],
        trainable=self.hparams['embedding_trainable'])

    embedded_sequences = embedding_layer(sequence_input)
    x = embedded_sequences
    
    x = self.get_gru_pool(x, l2_weight_decay)

  
    preds = Dense(6, activation='sigmoid', input_shape=(6,))(x)

    #rmsprop = RMSprop(lr=self.hparams['learning_rate'])
    #adam = Adam(lr=self.hparams['learning_rate'])
    nadam = Nadam(lr=self.hparams['learning_rate'])
    #self.model = Model(sequence_input, preds)
    #self.model.compile(
    #    loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    model = Model(sequence_input, preds)
    model.compile(
        loss='binary_crossentropy', optimizer=nadam, metrics=['acc'])
    return model
  
  def get_cnn(self, input_tensor, l2_weight_decay):
    for filter_size, kernel_size, pool_size in zip(
        self.hparams['cnn_filter_sizes'], self.hparams['cnn_kernel_sizes'],
        self.hparams['cnn_pooling_sizes']):
      output = self.build_conv_layer(input_tensor, filter_size, kernel_size, pool_size)
    output = Flatten()(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_weight_decay))(output)
    output = Dense(6, activation='sigmoid')(output)
    #output = Reshape(target_shape=(6,))(output)
    return output
  
  def get_lstm(self, input_tensor, l2_weight_decay, lstm_dim=50):
    output = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_tensor)
    output = Attention(self.hparams['max_sequence_length'])(output)
    output = Dense(128, activation='relu')(output)
    #output = GlobalMaxPooling1D()(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    
    #output = Dense(128, activation='relu')(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    #output = BatchNormalization()(output)
    output = Dense(6, activation='sigmoid')(output)
    
    return output
  
  def get_gru(self, input_tensor, l2_weight_decay, recurrent_units=512):
    output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(input_tensor)
    #output = Attention(self.hparams['max_sequence_length'])(output)
    #output = Dense(128, activation='relu')(output)
    #output = GlobalMaxPooling1D()(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(256, activation='linear')(output)
    output = LeakyReLU(alpha=.001)(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(128, activation='linear')(output)
    output = LeakyReLU(alpha=.001)(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(64, activation='linear')(output)
    output = LeakyReLU(alpha=.001)(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(32, activation='relu')(output)
    #output = LeakyReLU(alpha=.001)(output)
    #output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = BatchNormalization()(output)
    
    #output = Dense(6, activation='sigmoid')(output)
    
    return output
  def get_gru_pool(self, input_tensor, l2_weight_decay, recurrent_units=128):
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(input_tensor)
    att = Attention(self.hparams['max_sequence_length'])(x)
    att = Dropout(0.5)(att)
    avg_pool = GlobalAveragePooling1D()(x)
    avg_pool = Dropout(0.5)(avg_pool)
    max_pool = GlobalMaxPooling1D()(x)
    max_pool = Dropout(0.5)(max_pool)
    conc = concatenate([avg_pool, max_pool, att])
    output = Dropout(0.5)(conc)
    output = BatchNormalization()(output)
    #output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(input_tensor)
    #output = Attention(self.hparams['max_sequence_length'])(output)
    #output = Dense(128, activation='relu')(output)
    #output = GlobalMaxPooling1D()(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(256, activation='linear')(output)
    #output = LeakyReLU(alpha=.001)(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(128, activation='linear')(output)
    #output = LeakyReLU(alpha=.001)(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(64, activation='linear')(output)
    #output = LeakyReLU(alpha=.001)(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(32, activation='relu')(output)
    #output = LeakyReLU(alpha=.001)(output)
    #output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = BatchNormalization()(output)
    
    #output = Dense(6, activation='sigmoid')(output)
    
    return output
   
  def get_gru_lstm(self, input_tensor, l2_weight_decay, recurrent_units=64, lstm_dim=50):
    output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(input_tensor)
    #output = Attention(self.hparams['max_sequence_length'])(output)
    #output = Dense(128, activation='relu')(output)
    #output = GlobalMaxPooling1D()(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    #output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(output)
    output = GlobalMaxPooling1D()(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Dense(32, activation='relu')(output)
    #output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(output)
    #output = Dropout(self.hparams['dropout_rate'])(output)
    #output = BatchNormalization()(output)
    
    #output = Dense(6, activation='sigmoid')(output)
    
    return output
    
  def get_gru_two(self, input_tensor, l2_weight_decay, recurrent_units=64):
    output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(input_tensor)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(output)
    output = Dense(32, activation='relu')(output)

    
    return output
  
  def build_lstm_layer(self, input_tensor, lstm_dim=50):
    output = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_tensor)
    output = Dropout(self.hparams['dropout_rate'])(output)
    output = Bidirectional(LSTM((lstm_dim*2), return_sequences=True))(output)
    output = Dropout(self.hparams['dropout_rate'])(output)
    
    return output
    
  def multi_log_loss(self, y_true, y_pred):  # score function for CV
    # Handle all zeroes
    all_zeros = np.all(y_pred == 0, axis=1)
    y_pred[all_zeros] = 1/9
    # Normalise sum of row probabilities to one
    row_sums = np.sum(y_pred, axis=1)
    y_pred /= row_sums.reshape((-1, 1))
    # Calculate score
    n_rows = y_true.size
    y_true = y_true - 1  # classes start from 1 where columns start from zero
    score_sum = 0
    for i in range(y_true.size):
        score_sum -= np.log(y_pred[i, y_true[i]])
    score = score_sum / n_rows
    return score
    
  def calc_loss(y_true, y_pred):
    return np.mean([log_loss(y_true[:, i], y_pred[:, i]) 
                    for i in range(y_true.shape[1])])
  
  def _train_model(self, model, train_x, train_y, val_x, val_y, callbacks):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0
    #charCNN:LSTM
    #train_x = np.reshape(train_x, train_x.shape + (1,))
    #val_x = np.reshape(val_x, val_x.shape + (1,))
    
    while True:
        model.fit(
            train_x,
            train_y,
            batch_size=self.hparams['batch_size'],
            epochs=self.hparams['epochs'],
            validation_data=(val_x, val_y),
            callbacks=callbacks,
            verbose=2)
        
        y_pred = model.predict(val_x, batch_size=self.hparams['batch_size'])

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.

        print("Epoch {0} auc {1} best_auc {2}".format(current_epoch, total_loss, best_loss))
        

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model
  
  def train_leftover_folds(self, X, y, fold_count, model_list, fold):
      fold_size = len(X) // fold_count
      models = []
      total_meta = []
      for fold_id in range(0, fold_count):
          if (fold_id >= fold):
              fold_start = fold_size * fold_id
              fold_end = fold_start + fold_size
            
              if fold_id == fold_count - 1:
                  fold_end = len(X)

              train_x = np.concatenate([X[:fold_start], X[fold_end:]])
              train_y = np.concatenate([y[:fold_start], y[fold_end:]])

              val_x = X[fold_start:fold_end]
              val_y = y[fold_start:fold_end]
          
              save_path = os.path.join(self.model_dir, '%s_model.h5' % (self.model_name + str(fold_id)))
              callbacks = [
              ModelCheckpoint(
                  save_path, save_best_only=True, verbose=self.hparams['verbose'])
              ]

              model = self._train_model(model_list[fold_id], train_x, train_y, val_x, val_y,callbacks)
              meta = model_list[fold_id].predict(val_x, batch_size=128)
              if (fold_id == 0):
                  total_meta = meta
              else:
                  total_meta = np.concatenate((total_meta, meta), axis=0)
              model_path = os.path.join(self.model_dir, "model{0}_weights.npy".format(fold_id))
              np.save(model_path, model.get_weights())
              models.append(model)
          else:
              models.append(model_list[fold_id])

      return models

  def train_folds(self, X, y, fold_count, model_list):
      fold_size = len(X) // fold_count
      models = []
      total_meta = []
      for fold_id in range(0, fold_count):
          fold_start = fold_size * fold_id
          fold_end = fold_start + fold_size
            
          if fold_id == fold_count - 1:
              fold_end = len(X)

          train_x = np.concatenate([X[:fold_start], X[fold_end:]])
          train_y = np.concatenate([y[:fold_start], y[fold_end:]])

          val_x = X[fold_start:fold_end]
          val_y = y[fold_start:fold_end]
          
          save_path = os.path.join(self.model_dir, '%s_model.h5' % (self.model_name + str(fold_id)))
          callbacks = [
          ModelCheckpoint(
              save_path, save_best_only=True, verbose=self.hparams['verbose'])
          ]

          model = self._train_model(model_list[fold_id], train_x, train_y, val_x, val_y,callbacks)
          meta = model.predict(val_x, batch_size=128)
          if (fold_id == 0):
              total_meta = meta
          else:
              total_meta = np.concatenate((total_meta, meta), axis=0)
          model_path = os.path.join(self.model_dir, "model{0}_weights.npy".format(fold_id))
          np.save(model_path, model.get_weights())
          models.append(model)

      return models, total_meta

  def predict(self, texts):
    """Returns model predictions on texts."""
    data = self.prep_text(texts)
    return self.model.predict(data)

  def score_auc(self, texts, labels):
    preds = self.predict(texts)
    return compute_auc(labels, preds)
  
  def score_pred(self, texts, labels):
    preds = self.predict(texts)
    return preds

  def summary(self):
    return self.model.summary()

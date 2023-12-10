from utils import get_filtered_questions, clean_no_stopwords, get_data_for_cognitive_classifiers, get_glove_vectors
from mnbc import MNBC
from svm_glove import TfidfEmbeddingVectorizer, load_svm_model
from brnn import BiDirectionalRNN, RNN, relu, relu_prime, sent_to_glove, clip, load_brnn_model
# from rnn import *
from blstm import *
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from svm_glove import TfidfEmbeddingVectorizer
from maxent import features
from sklearn.model_selection import train_test_split
import csv
import re
import numpy as np
import pickle
import random
import brnn
import os
import platform
import sys
# from sklearn.externals import joblib
import dill as pickle
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

domain = pickle.load(
    open(os.path.join(os.path.dirname(__file__), 'resources/domain.pkl'),  'rb'))
domain = {k: set(clean_no_stopwords(
    ' '.join(list(domain[k])), stem=False)) for k in domain.keys()}
domain_names = domain.keys()

keywords = set()
for k in domain:
    keywords = keywords.union(
        set(list(map(str.lower, map(str, list(domain[k]))))))

mapping_cog = {'Remember': 0, 'Understand': 1,
               'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog2 = {v: k for k, v in mapping_cog.items()}

# transformation for BiRNN. This should actually become a part of the RNN for better code maintainability
NUM_CLASSES = 6

''' these params are for BCL models '''
VEC_SIZE_SVM = 100
VEC_SIZE_BRNN = 100
CUSTOM_GLOVE_SVM = False
CUSTOM_GLOVE_BRNN = False

''' these params are for normal models '''
'''
VEC_SIZE_SVM = 100
VEC_SIZE_BRNN = 300
CUSTOM_GLOVE_SVM = False
CUSTOM_GLOVE_BRNN = True
'''

savepath = 'glove.%dd%s.pkl' % (
    VEC_SIZE_SVM, '_custom' if CUSTOM_GLOVE_SVM else '')
print(savepath)

svm_w2v = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'resources/GloVe/' + savepath), 'rb'))

savepath = 'glove.%dd%s.pkl' % (
    VEC_SIZE_BRNN, '_custom' if CUSTOM_GLOVE_BRNN else '')
brnn_w2v = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'resources/GloVe/' + savepath), 'rb'))

print('Loaded GloVe models')

####################### ONE TIME MODEL LOADING #########################


def get_cog_models(get_ann=True):
    if platform.system() == 'Windows':
        suffix = '_windows'
    else:
        suffix = ''

    ################# BRNN MODEL #################
    clf_brnn = load_brnn_model('brnn_model.pkl', brnn_w2v)
    print('Loaded BiRNN model')

    ################# SVM-GLOVE MODEL #################
    print('glove_svm_model_bcl%s.pkl' % suffix)
    clf_svm = load_svm_model('glove_svm_model.pkl', svm_w2v)
    print('Loaded SVM-GloVe model')

    ################# BLSTM MODEL #################
    clf_lstm = joblib.load(os.path.join(
        os.path.dirname(__file__), 'models/BiLSTM/blstm_model.pkl'))
    print('Loaded BLSTM model')

    ################# MNBC MODEL #################
    clf_mnbc = joblib.load(os.path.join(
        os.path.dirname(__file__), 'models/MNBC/mnbc.pkl'))
    print('Loaded MNBC model')

    ################# MLP MODEL #################
    nn = None
    if get_ann:
        nn = joblib.load(os.path.join(os.path.dirname(
            __file__), 'models/cog_ann_voter_bcl.pkl'))
        print('Loaded MLP model')

    return clf_mnbc, clf_brnn, clf_lstm, nn

##################### PREDICTION WITH PARAMS ############################


def predict_cog_label(question, models, subject='ADA'):
    clf_mnbc, clf_brnn, clf_lstm, nn = models
    question2 = get_filtered_questions(
        question, threshold=0.15, what_type=subject.lower())  # svm and birnn
    if len(question2) > 0:
        question = question2[0]
    X1 = np.array(question.split()).reshape(1, -1)

    # softmax probabilities

    # probs_svm, probs_mnbc, probs_brnn = get_model_probs(X1, models)

    probs_mnbc, probs_brnn, probs_lstm = get_model_probs(X1, models)
    X = np.hstack((probs_brnn, probs_mnbc)).reshape(
        1, -1)  # concatenating the vectors
    print(X)
    return nn.predict(X), nn.predict_proba(X)

##################### PREDICTION HELPER PARAMS ############################


def get_model_probs(X, models):
    # clf_svm, clf_mnbc, clf_brnn, nn = models
    clf_mnbc, clf_brnn, clf_lstm, nn = models
    # print(clf_svm)
    # print(clf_mnbc)
    # print(clf_brnn)
    probs_lstm = clf_lstm.predict(X)
    for i in range(len(probs_lstm)):
        probs = probs_lstm[i]
        probs_lstm[i] = np.exp(probs) / np.sum(np.exp(probs))
    probs_lstm = np.array(probs_lstm)

    probs_mnbc = clf_mnbc.predict_proba(X)
    # print(probs_mnbc)
    for i in range(len(probs_mnbc)):
        probs = probs_mnbc[i]
        probs_mnbc[i] = np.exp(probs) / np.sum(np.exp(probs))
    probs_mnbc = np.array(probs_mnbc)

    probs_brnn = clf_brnn.predict_proba(X)
    for i in range(len(probs_brnn)):
        probs = probs_brnn[i]
        probs_brnn[i] = np.exp(probs) / np.sum(np.exp(probs))
    probs_brnn = np.array(probs_brnn)

    # return probs_svm, probs_mnbc, probs_brnn
    return probs_mnbc, probs_brnn, probs_lstm


#########################################################################
#                            MAIN BEGINS HERE                           #
#########################################################################
if __name__ == '__main__':

    ################ MODEL LOADING ##################
    models = get_cog_models(get_ann=False)
    clf_mnbc, clf_brnn, clf_lstm, _ = models

    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.20, 0.20],
                                                          what_type=[
                                                              'ada', 'bcl', 'os'],
                                                          include_keywords=True,
                                                          keep_dup=False)

    X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.20],
                                                        what_type=[
                                                            'ada', 'bcl', 'os'],
                                                        what_for='test',
                                                        include_keywords=False,
                                                        keep_dup=False)

    # softmax probabilities
    ptrain_mnbc, ptrain_brnn, ptrain_lstm = get_model_probs(X_train, models)
    ptest_mnbc, ptest_brnn, ptest_lstm = get_model_probs(X_test, models)

    print('Loaded data for voting system')

    # concatenating the vectors
    X_train = np.hstack((ptrain_brnn, ptrain_mnbc, ptrain_lstm))
    X_test = np.hstack((ptest_brnn, ptest_mnbc, ptest_lstm)
                       )  # concatenating the vectors

    ###### NEURAL NETWORK BASED VOTING SYSTEM ########
    clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(128, 16),
                        batch_size=4, learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
    print(X_train, Y_train)
    clf.fit(X_train, Y_train)
    print('ANN training completed')
    Y_real, Y_pred = Y_test, clf.predict(X_test)

    joblib.dump(clf, os.path.join(os.path.dirname(
        __file__), 'models/cog_ann_voter.pkl'))

    print('Accuracy: {:.2f}%'.format(accuracy_score(Y_real, Y_pred) * 100))

    y_pred_lstm = []
    y_pred_mnbc = []
    y_pred_brnn = []

    for x in X_test:
        y_pred_lstm.append(np.argmax(x[:6]))
        y_pred_mnbc.append(np.argmax(x[6:12]))
        y_pred_brnn.append(np.argmax(x[12:]))

    print(
        'SVM-GloVe Accuracy: {:.2f}%'.format(accuracy_score(Y_real, y_pred_lstm) * 100))
    print('MNBC Accuracy: {:.2f}%'.format(
        accuracy_score(Y_real, y_pred_mnbc) * 100))
    print('BiRNN Accuracy: {:.2f}%'.format(
        accuracy_score(Y_real, y_pred_brnn) * 100))
'''


'''

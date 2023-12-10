from keras.models import Sequential
import os
import sys
import pickle
import joblib
import dill
import numpy as np
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import sequence
from tensorflow.keras.utils import pad_sequences
from utils import get_data_for_cognitive_classifiers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sys.setrecursionlimit(2 * 10 ** 7)

NUM_CLASSES = 6

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

INPUT_SIZE = 300
filename = 'glove.6B.%dd.txt' % INPUT_SIZE


def clip(v):
    return v[:10]


mapping_cog = {'Remember': 0, 'Understand': 1,
               'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
domain = pickle.load(
    open(os.path.join(os.path.dirname(__file__), 'resources/domain.pkl'),  'rb'))

if not os.path.exists('resources/GloVe/%s_saved.pkl' % filename.split('.txt')[0]):
    print()
    with open('models/' + filename, "r", encoding='utf-8') as lines:
        w2v = {}
        for row, line in enumerate(lines):
            try:
                w = line.split()[0]
                if w not in vocabulary:
                    continue
                vec = np.array(list(map(float, line.split()[1:])))
                w2v[w] = vec
            except:
                continue
            finally:
                print(CURSOR_UP_ONE + ERASE_LINE +
                      'Processed {} GloVe vectors'.format(row + 1))

    dill.dump(w2v, open('models/%s_saved.pkl' %
              filename.split('.txt')[0], 'wb'))
else:
    w2v = dill.load(open('resources/GloVe/%s_saved.pkl' %
                    filename.split('.txt')[0], 'rb'))


def sent_to_glove(questions):
    questions_w2glove = []

    for question in questions:
        vec = []
        for word in question[:10]:
            if word in w2v:
                vec.append(w2v[word])
            else:
                vec.append(np.zeros(len(w2v['the'])))
        questions_w2glove.append(np.array(vec))

    return np.array(questions_w2glove)


class SkillClassifier:
    def __init__(self, input_dim=300, hidden_dim=128, dropout=0.2):
        np.random.seed(7)

        '''
        self.model.add(LSTM(hidden_dims[0], input_shape=(None, input_dim), return_sequences=True, recurrent_dropout=dropout))
        self.model.add(LSTM(hidden_dims[1], dropout=dropout))
        self.model.add(Dense(NUM_CLASSES, kernel_initializer="lecun_uniform", activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
        '''

        self.model = Sequential()
        # self.model.add(Embedding())
        self.model.add(Bidirectional(LSTM(hidden_dim, input_shape=(
            None, input_dim), recurrent_dropout=dropout)))
        self.model.add(
            Dense(NUM_CLASSES, kernel_initializer="lecun_uniform", activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train(self, train_data, val_data, epochs=5, batch_size=64):
        # print(self.model.summary())
        history = self.model.fit(train_data[0], train_data[1], epochs=epochs, shuffle=True,
                                 batch_size=batch_size, validation_data=(val_data[0], val_data[1]))

        return history

    def test(self, test_data):
        y_pred = self.model.predict(test_data[0]).round()
        print("Accuracy : ", accuracy_score(test_data[1], y_pred))
        print(classification_report(test_data[1], y_pred))
        # cm = confusion_matrix(test_data[1].argmax(
        #     axis=1), y_pred.argmax(axis=1))
        # sns.heatmap(cm,
        #             annot=True,
        #             fmt='g')

        # plt.ylabel('Prediction', fontsize=13)
        # plt.xlabel('Actual', fontsize=13)
        # plt.title('Confusion Matrix', fontsize=17)
        # plt.show()
        return self.model.evaluate(test_data[0], test_data[1], verbose=0)[1]

    def save(self):
        self.model.save('models/rnn_model.h5')

    def __prep_data(self, X):
        if type(X[0]) != str:
            return [' '.join(x) for x in X]

        return X

    def predict(self, X):
        return self.model.predict(self.__prep_data(X))


if __name__ == "__main__":
    X_data = []
    Y_data = []

    clf = SkillClassifier(input_dim=len(w2v['the']))

    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.1, 0.15, 0.2, 0.25], what_type=[
        'ada', 'bcl', 'os'], include_keywords=True, keep_dup=False)
    X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.1, 0.15, 0.2, 0.25], what_type=[
        'ada', 'bcl', 'os'], include_keywords=True, keep_dup=False)

    X_data = X_train + X_test
    Y_data = Y_train + Y_test

    X_data = tf.keras.preprocessing.sequence.pad_sequences(
        sent_to_glove(X_data), maxlen=10)

    for i in range(len(Y_data)):
        v = np.zeros(NUM_CLASSES)
        v[Y_data[i]] = 1
        Y_data[i] = v

    # Y_data = np.array(Y_data)
    X_data = np.array(X_data).astype(np.float32)
    Y_data = np.array(Y_data).astype(np.float32)

    X_train = np.array(X_data[: int(len(X_data) * 0.70)])
    Y_train = np.array(Y_data[: int(len(X_data) * 0.70)])

    X_val = np.array(X_data[int(len(X_data) * 0.70): int(len(X_data) * 0.8)])
    Y_val = np.array(Y_data[int(len(X_data) * 0.70): int(len(X_data) * 0.8)])

    X_test = np.array(X_data[int(len(X_data) * 0.8):])
    Y_test = np.array(Y_data[int(len(X_data) * 0.8):])

    history = clf.train(train_data=(X_train, Y_train), val_data=(
        X_val, Y_val), epochs=50, batch_size=4)
    joblib.dump(clf, os.path.join(
        os.path.dirname(__file__), 'models/BiLSTM/blstm_model.pkl'))
    print(str(clf.test(test_data=(X_test, Y_test)) * 100)[:5] + '%')
    clf.save()

    # Plotting learning behavior graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'],
             label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'],
             label='Validation Accuracy', color='green')
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'],
             label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Model Learning Behavior with Different Number of Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

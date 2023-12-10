import os
import sys
import random
import pickle
import signal
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_data_for_cognitive_classifiers, get_glove_vectors

sys.setrecursionlimit(2 * 10 ** 7)


def sent_to_glove(questions, w2v):
    questions_w2glove = []

    for question in questions:
        vec = []
        for word in question:
            if word in w2v:
                vec.append(w2v[word])
            else:
                vec.append(np.zeros(300))
        questions_w2glove.append(np.array(vec))

    return np.array(questions_w2glove)


def relu(z):
    y = z * (z > 0)
    return np.clip(y, 0, 5)


def relu_prime(z):
    return (z > 0)


def clip(v):
    return v[:10]


class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, direction="right"):
        self.direction = direction
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.f = relu  # np.tanh
        self.f_prime = relu_prime  # lambda x: 1 - (x ** 2)

        self.Wxh = np.random.randn(
            hidden_size, input_size) * np.sqrt(2.0 / (hidden_size + input_size))
        self.Whh = np.random.randn(
            hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size * 2))
        self.Why = np.random.randn(
            output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.bh = np.zeros((hidden_size, 1))
        # output bias - computed but not used
        self.by = np.zeros((output_size, 1))

        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(
            self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(
            self.by)  # memory variables for Adagrad

        self.dropout_percent = 0.2

    def forward(self, x, hprev, do_dropout=False):
        if (self.direction == 'left'):
            x = x[::-1]

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)

        seq_length = len(x)

        for t in range(seq_length):
            xs[t] = x[t].reshape(-1, 1)
            hs[t] = self.f(np.dot(self.Wxh, xs[t]) +
                           np.dot(self.Whh, hs[t-1]) + self.bh)

            if (do_dropout):
                hs[t] *= np.random.binomial(1, 1 -
                                            self.dropout_percent, size=hs[t-1].shape)

            ys[t] = self.f(np.dot(self.Why, hs[t]) + self.by)
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        return xs, hs, ys, ps

    def backprop(self, xs, hs, ys, ps, targets, dy, do_dropout=False):
        if (self.direction == 'left'):
            xs = {len(xs) - 1 - k: xs[k] for k in xs}

        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(
            self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[-1])

        for t in reversed(range(len(xs))):
            tmp = dy[t] * self.f_prime(ys[t])
            dWhy += np.dot(tmp, hs[t].T)
            dby += tmp
            dh = np.dot(self.Why.T, dy[t]) + dhnext
            dhraw = dh * self.f_prime(hs[t])
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            # clip to mitigate exploding gradients
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby, hs[len(xs) - 1]

    def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / \
                np.sqrt(mem + 1e-8)  # adagrad update


class BiDirectionalRNN:
    def __init__(self, w2v, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.right = RNN(input_size, hidden_size, output_size,
                         learning_rate, direction="right")
        self.left = RNN(input_size, hidden_size, output_size,
                        learning_rate, direction="left")

        self.by = np.zeros((output_size, 1))
        self.mby = np.zeros_like(self.by)

        self.w2v = w2v

    def forward(self, x):
        seq_length = len(x)

        y_pred = []
        dby = np.zeros_like(self.by)
        xsl, hsl, ysl, psl = self.left.forward(
            x, np.zeros((self.hidden_size, 1)))
        xsr, hsr, ysr, psr = self.right.forward(
            x, np.zeros((self.hidden_size, 1)))

        for ind in range(seq_length):
            this_y = np.dot(
                self.right.Why, hsr[ind]) + np.dot(self.left.Why, hsl[ind]) + self.by
            y_pred.append(this_y)

        return np.argmax(y_pred[-1])

    def __sent_to_glove(self, X):
        return sent_to_glove(X, self.w2v)

    def fit(self, X, Y, validation_data=None, epochs=5, do_dropout=False):
        X = self.__sent_to_glove(X)

        for e in range(epochs):
            print('Epoch {}'.format(e + 1))

            data = list(zip(X, Y))
            random.shuffle(data)
            X = [x[0] for x in data]
            Y = [x[1] for x in data]
            for x, y in zip(X, Y):
                x = clip(x)

                hprevr = np.zeros((self.hidden_size, 1))
                hprevl = np.zeros((self.hidden_size, 1))

                seq_length = len(x)

                xsl, hsl, ysl, psl = self.left.forward(x, hprevl, do_dropout)
                xsr, hsr, ysr, psr = self.right.forward(x, hprevr, do_dropout)

                y_pred = []
                dy = []
                dby = np.zeros_like(self.by)
                for ind in range(seq_length):
                    this_y = np.dot(
                        self.right.Why, hsr[ind]) + np.dot(self.left.Why, hsl[ind]) + self.by
                    y_pred.append(this_y)

                for ind in range(seq_length):
                    this_dy = np.exp(y_pred[ind]) / np.sum(np.exp(y_pred[ind]))
                    t = np.argmax(y)
                    this_dy[t] -= 1
                    dy.append(this_dy)
                    dby += this_dy

                y_pred = np.array(y_pred)
                dy = np.array(dy)

                self.mby += dby * dby
                self.by += -self.learning_rate * dby / \
                    np.sqrt(self.mby + 1e-8)  # adagrad update

                dWxhr, dWhhr, dWhyr, dbhr, dbyr, hprevr = self.right.backprop(
                    xsr, hsr, ysr, psr, y, dy, do_dropout)
                dWxhl, dWhhl, dWhyl, dbhl, dbyl, hprevl = self.left.backprop(
                    xsl, hsl, ysl, psl, y, dy, do_dropout)

                self.right.update_params(dWxhr, dWhhr, dWhyr, dbhr, dbyr)
                self.left.update_params(dWxhl, dWhhl, dWhyl, dbhl, dbyl)

            if validation_data is not None:
                self.get_accuracy_score(validation_data[0], validation_data[1])

        print("\nTraining done.")

        return self

    def predict(self, X):
        X = self.__sent_to_glove(X)
        predictions = []
        for x in X:
            x = clip(x)
            predictions.append(self.forward(x))

        return np.array(predictions)

    def predict_proba(self, X):
        X = self.__sent_to_glove(X)
        prob_predictions = []
        for x in X:
            seq_length = len(x)

            y_pred = []
            dby = np.zeros_like(self.by)
            xsl, hsl, ysl, psl = self.left.forward(
                x, np.zeros((self.hidden_size, 1)))
            xsr, hsr, ysr, psr = self.right.forward(
                x, np.zeros((self.hidden_size, 1)))

            for ind in range(seq_length):
                this_y = np.dot(
                    self.right.Why, hsr[ind]) + np.dot(self.left.Why, hsl[ind]) + self.by
                y_pred.append(this_y)

            prob_predictions.append(y_pred[-1].reshape(-1,))

        return prob_predictions

    def get_accuracy_score(self, X, Y, test=False):
        X = self.__sent_to_glove(X)
        targets = []
        predictions = []
        for x, y in zip(X, Y):
            x = clip(x)
            tr = np.argmax(y)
            op = self.forward(x)
            targets.append(tr)
            predictions.append(op)

        if not test:
            print('[val acc:      {:.2f}%]'.format(
                accuracy_score(targets, predictions) * 100))
            print('[val f1 score: {:.2f}]'.format(
                f1_score(targets, predictions, average="macro")))
            print(precision_score(targets, predictions, average="macro"))
            print(recall_score(targets, predictions, average="macro"))
        else:
            print(classification_report(targets, predictions))
            cm = confusion_matrix(targets, predictions)
            sns.heatmap(cm,
                        annot=True,
                        fmt='g')

            plt.ylabel('Prediction', fontsize=13)
            plt.xlabel('Actual', fontsize=13)
            plt.title('Confusion Matrix', fontsize=17)
            plt.show()

        return accuracy_score(targets, predictions)


def save_brnn_model(clf):
    params = clf.input_size, clf.hidden_size, clf.output_size, clf.learning_rate, clf.right, clf.left, clf.by, clf.mby
    with open(os.path.join(os.path.dirname(__file__), 'models/BiRNN/brnn_model.pkl'), 'wb') as f:
        pickle.dump(params, f)


def load_brnn_model(model_name, w2v):
    with open(os.path.join(os.path.dirname(__file__), 'models/BiRNN/' + model_name), 'rb') as f:
        params = pickle.load(f)
        clf = BiDirectionalRNN(w2v, params[0], params[1], params[2], params[3])
        clf.right = params[4]
        clf.left = params[5]
        clf.by = params[6]
        clf.mby = params[7]

    return clf


if __name__ == "__main__":
    NUM_CLASSES = 6
    INPUT_SIZE = 300
    USE_CUSTOM_GLOVE_MODELS = True
    TRAIN = True
    RETRAIN = True

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'

    vocabulary = {'the'}

    if USE_CUSTOM_GLOVE_MODELS:
        savepath = 'glove.%dd_custom.pkl' % INPUT_SIZE
    else:
        savepath = 'glove.%dd.pkl' % INPUT_SIZE

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath)):
        w2v = {}
        if USE_CUSTOM_GLOVE_MODELS:
            print('Loading custom vectors')
            print()
            w2v.update(get_glove_vectors('resources/GloVe/' +
                       'glove.ADA.%dd.txt' % INPUT_SIZE))
            print()
            w2v.update(get_glove_vectors('resources/GloVe/' +
                       'glove.OS.%dd.txt' % INPUT_SIZE))

        print()
        w2v.update(get_glove_vectors('resources/GloVe/' +
                   'glove.6B.%dd.txt' % INPUT_SIZE))
        pickle.dump(w2v, open(os.path.join(os.path.dirname(
            __file__), 'resources/GloVe/' + savepath), 'wb'))
    else:
        w2v = pickle.load(open(os.path.join(os.path.dirname(
            __file__), 'resources/GloVe/' + savepath), 'rb'))
    print('Loaded Glove w2v')

    X_data = []
    Y_data = []

    #X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.2, 0.25, 0.3, 0.35], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=True, keep_dup=False)

    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.25, 0.25],
                                                          what_type=[
                                                              'ada', 'os', 'bcl'],
                                                          include_keywords=True,
                                                          keep_dup=False)
    print(len(X_train))

    X_test1, Y_test1 = get_data_for_cognitive_classifiers(threshold=[0.25],
                                                          what_type=[
                                                              'ada', 'os', 'bcl'],
                                                          what_for='test',
                                                          keep_dup=False)
    for i in range(len(Y_train)):
        v = np.zeros(NUM_CLASSES)
        v[Y_train[i]] = 1
        Y_train[i] = v

    for i in range(len(Y_test1)):
        v = np.zeros(NUM_CLASSES)
        v[Y_test1[i]] = 1
        Y_test1[i] = v

    X_test = np.array(X_test1[: int(len(X_test1) * 0.8)])
    Y_test = np.array(Y_test1[: int(len(X_test1) * 0.8)])

    X_val = np.array(X_test1[int(len(X_test1) * 0.8):])
    Y_val = np.array(Y_test1[int(len(X_test1) * 0.8):])

    print('Data Loaded/Preprocessed')

    HIDDEN_SIZE = 200
    OUTPUT_SIZE = NUM_CLASSES

    EPOCHS = 2
    LEARNING_RATE = 0.010

    BRNN = None
    if TRAIN:
        if RETRAIN:
            BRNN = load_brnn_model('brnn_model.pkl', w2v=w2v)
        else:
            BRNN = BiDirectionalRNN(
                w2v, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)

        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            BRNN.fit(X_train, Y_train, validation_data=(
                X_val, Y_val), epochs=EPOCHS, do_dropout=False)
        except KeyboardInterrupt:
            print('\tTraining stopped: keyboard interrupt')

        save_brnn_model(BRNN)
    else:
        BRNN = load_brnn_model('brnn_model.pkl', w2v=w2v)

    print()
    accuracy = BRNN.get_accuracy_score(X_test, Y_test, True)

    print("Accuracy: {:.2f}%".format(accuracy * 100))

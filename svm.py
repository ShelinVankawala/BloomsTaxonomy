import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from utils import get_data_for_cognitive_classifiers

# Mapping of cognitive levels to numerical labels
mapping_cog = {'Remember': 0, 'Understand': 1,
               'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}

# Load your dataset, assuming you have a DataFrame with 'question' and 'cognitive_level' columns
# Replace 'your_dataset.csv' with the actual path to your dataset

# Split the dataset into training and testing sets
X_train, y_train = get_data_for_cognitive_classifiers(threshold=[0.10, 0.10],
                                                      what_type=[
                                                          'ada', 'os', 'bcl'],
                                                      include_keywords=True,
                                                      keep_dup=False)
print(len(X_train))

X_test, y_test = get_data_for_cognitive_classifiers(threshold=[0.10],
                                                    what_type=[
                                                        'ada', 'os', 'bcl'],
                                                    what_for='test',
                                                    keep_dup=False)

X_train = [' '.join(x) for x in X_train]
X_test = [' '.join(x) for x in X_test]

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# SVM model
# You can tune hyperparameters as needed
svm_classifier = SVC(kernel='linear', C=1)

# Fit the SVM model on the training data
svm_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Binarize the labels for multi-class ROC curves
y_test_bin = label_binarize(y_test, classes=list(range(6)))

# Predict decision function scores
decision_values = svm_classifier.decision_function(X_test_tfidf)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], decision_values[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i in range(6):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for SVM Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()

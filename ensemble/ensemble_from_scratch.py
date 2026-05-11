import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD AND SAMPLE DATA
# Using a sample of 20,000 to keep 'from scratch' execution times reasonable
df = pd.read_csv('phishing_site_urls.csv')
df = df.sample(n=20000, random_state=42) 

# 2. PREPROCESSING
le = LabelEncoder()
y = le.fit_transform(df['Label']) # 'bad' -> 1, 'good' -> 0

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf.fit_transform(df['URL']).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ENSEMBLE CLASSES (BAGGING & BOOSTING)
class PhishingEnsemble:
    def __init__(self, method='bagging', n_estimators=10):
        self.method = method
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = [] 

    def fit(self, X, y):
        n_samples = X.shape[0]
        if self.method == 'bagging':
            for _ in range(self.n_estimators):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                model = DecisionTreeClassifier(max_depth=10)
                model.fit(X[indices], y[indices])
                self.models.append(model)
        elif self.method == 'boosting':
            w = np.full(n_samples, (1 / n_samples))
            for _ in range(self.n_estimators):
                model = DecisionTreeClassifier(max_depth=1)
                model.fit(X, y, sample_weight=w)
                preds = model.predict(X)
                error = np.sum(w[preds != y]) / np.sum(w)
                alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
                w *= np.exp(alpha * (preds != y).astype(int) - alpha * (preds == y).astype(int))
                w /= np.sum(w)
                self.models.append(model)
                self.alphas.append(alpha)

    def predict(self, X):
        if self.method == 'bagging':
            all_preds = np.array([m.predict(X) for m in self.models])
            return np.array([np.argmax(np.bincount(all_preds[:, i])) for i in range(X.shape[0])])
        elif self.method == 'boosting':
            final_effort = np.zeros(X.shape[0])
            for alpha, model in zip(self.alphas, self.models):
                p = np.where(model.predict(X) == 0, -1, 1)
                final_effort += alpha * p
            return np.where(final_effort >= 0, 1, 0)

# 4. STACKING CLASS
class StackingFromScratch:
    def __init__(self):
        self.base_models = [DecisionTreeClassifier(max_depth=5), LogisticRegression()]
        self.meta_model = LogisticRegression()

    def fit(self, X, y):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            meta_features[:, i] = model.predict(X)
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)
        return self.meta_model.predict(meta_features)

# 5. RUN EVALUATION
models_to_test = {
    "Bagging (Random Forest Style)": PhishingEnsemble(method='bagging', n_estimators=10),
    "Boosting (AdaBoost Style)": PhishingEnsemble(method='boosting', n_estimators=20),
    "Stacking (Meta-Learner)": StackingFromScratch()
}

for name, model in models_to_test.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))


'''
--- Training Bagging (Random Forest Style) ---
              precision    recall  f1-score   support

           0       0.83      0.64      0.72      1163
           1       0.86      0.95      0.90      2837

    accuracy                           0.86      4000
   macro avg       0.85      0.79      0.81      4000
weighted avg       0.86      0.86      0.85      4000


--- Training Boosting (AdaBoost Style) ---
              precision    recall  f1-score   support

           0       0.75      0.32      0.45      1163
           1       0.77      0.96      0.86      2837

    accuracy                           0.77      4000
   macro avg       0.76      0.64      0.65      4000
weighted avg       0.77      0.77      0.74      4000


--- Training Stacking (Meta-Learner) ---
              precision    recall  f1-score   support

           0       0.93      0.69      0.79      1163
           1       0.89      0.98      0.93      2837

    accuracy                           0.89      4000
   macro avg       0.91      0.83      0.86      4000
weighted avg       0.90      0.89      0.89      4000
'''
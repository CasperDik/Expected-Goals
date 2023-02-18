import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("input_data.csv")
X = data.drop(["outcome", "player_name", "x", "y"], axis=1)
y = data["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
{
    "label": "Logistic Regression",
    "model": LogisticRegression(),
},
{
    "label": "Random Forest",
    "model": RandomForestClassifier(max_depth=2),
},
{
    "label": "Support Vector Machine",
    "model": svm.SVC(probability=True),
},
{
    "label": "Multilayered Perceptron",
    "model": MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2)),
},
{
    "label": "XGBoost",
    "model": xgb.XGBClassifier(objective="binary:logistic"),
}
]

for m in models:
    model = m["model"]
    model.fit(X_train, y_train)
    xg = model.predict_proba(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, xg[:, 1])
    auc = metrics.roc_auc_score(y_test, model.predict(X_test))
    print("-" * 10, m["label"], "-" * 10)
    print(model.score(X_test, y_test))
    print(auc)
    plt.plot(fpr, tpr, label=m["label"])

plt.legend()
plt.show()

pickle.dump(models[4]["model"], open('finalized_model.sav', 'wb'))


# todo: add other metrics
# todo: optimize performance models
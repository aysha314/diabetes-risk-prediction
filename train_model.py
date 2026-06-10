import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df=pd.read_csv("cleaned_diabetes.csv")
print(df.head())

X=df.drop("Outcome",axis=1)
y=df["Outcome"]

print(X.head())
print(y.head())

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(X_train.shape)
print(X_test.shape)
    
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print("feature scaling completed")

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr.fit(X_train_scaled,y_train)
lr_pred=lr.predict(X_test_scaled)


rf=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n", "="*40)
    print(name)
    print("="*40)

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    return accuracy, precision, recall, f1

lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred
)

rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

if rf_f1 > lr_f1:
    best_model = rf
    best_model_name = "Random Forest"
else:
    best_model = lr
    best_model_name = "Logistic Regression"


joblib.dump(best_model, "diabetes_model.pkl")

print("\nBest Model Selected:")
print(best_model_name)

print("\nModel Saved:")
print("diabetes_model.pkl")



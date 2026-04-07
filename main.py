
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

df = pd.read_csv("loan_data.csv") 
#convert categorical to nmerical
df = pd.get_dummies(df, drop_first=True)

# Fill missing values
df = df.fillna(df.mean())

# Separate features and target
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

# Save column structure
columns = list(X.columns)
with open("columns.json", "w") as f:
    json.dump(columns, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = X_train.copy()
train_data["target"] = y_train

approved = train_data[train_data["target"] == 1]
rejected = train_data[train_data["target"] == 0]

# Downsample rejected class
rejected_downsampled = resample(
    rejected,
    replace=False,
    n_samples=int(len(rejected) * 0.3),
    random_state=42
)

imbalanced_data = pd.concat([approved, rejected_downsampled])

X_train_imb = imbalanced_data.drop("target", axis=1)
y_train_imb = imbalanced_data["target"]

#train model

lr = LogisticRegression(max_iter=5000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

#imbalanced model
lr_imb = LogisticRegression(max_iter=5000)
dt_imb = DecisionTreeClassifier()
rf_imb = RandomForestClassifier()

lr_imb.fit(X_train_imb, y_train_imb)
dt_imb.fit(X_train_imb, y_train_imb)
rf_imb.fit(X_train_imb, y_train_imb)

# mitigated models
lr_bal = LogisticRegression(class_weight='balanced', max_iter=5000)
dt_bal = DecisionTreeClassifier(class_weight='balanced')
rf_bal = RandomForestClassifier(class_weight='balanced')

lr_bal.fit(X_train, y_train)
dt_bal.fit(X_train, y_train)
rf_bal.fit(X_train, y_train)

#saving the models
models = {
    "lr": lr,
    "dt": dt,
    "rf": rf,
    "lr_imb": lr_imb,
    "dt_imb": dt_imb,
    "rf_imb": rf_imb,
    "lr_bal": lr_bal,
    "dt_bal": dt_bal,
    "rf_bal": rf_bal
}

with open("all_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("✅ Models trained and saved!")


def predict_all(input_data):
    """
    input_data = list of feature values in correct column order
    """

    # Load models
    with open("all_models.pkl", "rb") as f:
        models = pickle.load(f)

    input_array = np.array(input_data).reshape(1, -1)

    # Add noise
    noisy_input = input_array + np.random.normal(0, 0.3, input_array.shape)

    results = {}

    # NORMAL
    results["Logistic_Normal"] = int(models["lr"].predict(input_array)[0])
    results["Tree_Normal"] = int(models["dt"].predict(input_array)[0])
    results["RF_Normal"] = int(models["rf"].predict(input_array)[0])

    # IMBALANCED
    results["Logistic_Imbalanced"] = int(models["lr_imb"].predict(input_array)[0])
    results["Tree_Imbalanced"] = int(models["dt_imb"].predict(input_array)[0])
    results["RF_Imbalanced"] = int(models["rf_imb"].predict(input_array)[0])

    # MITIGATED
    results["Logistic_Mitigated"] = int(models["lr_bal"].predict(input_array)[0])
    results["Tree_Mitigated"] = int(models["dt_bal"].predict(input_array)[0])
    results["RF_Mitigated"] = int(models["rf_bal"].predict(input_array)[0])

    # NOISE TEST
    results["Logistic_Noisy"] = int(models["lr"].predict(noisy_input)[0])
    results["Tree_Noisy"] = int(models["dt"].predict(noisy_input)[0])
    results["RF_Noisy"] = int(models["rf"].predict(noisy_input)[0])

    return results


# Example: random test input
sample = X_test.iloc[0].tolist()
print("\n🔍 Sample Prediction:\n", predict_all(sample))

import matplotlib
matplotlib.use("TkAgg")

print("✅ Script started running...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore")

# ========== 1. LOAD DATA ==========
df = pd.read_csv("Tesla.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# ========== 2. STOCK CLOSE PRICE GRAPH ==========
plt.figure(figsize=(15, 5))
plt.plot(df["Close"])
plt.title("Tesla Close Price")
plt.ylabel("Price")
plt.show()

# ========== 3. FEATURE ENGINEERING ==========
splitted = df["Date"].str.split("/", expand=True)
df["day"] = splitted[1].astype("int")
df["month"] = splitted[0].astype("int")
df["year"] = splitted[2].astype("int")

df["is_quarter_end"] = np.where(df["month"] % 3 == 0, 1, 0)
df["open-close"] = df["Open"] - df["Close"]
df["low-high"] = df["Low"] - df["High"]
df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

# ========== 4. DATA PREPARATION ==========
features = df[["open-close", "low-high", "is_quarter_end"]]
target = df["target"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features_scaled, target, test_size=0.1, random_state=2022
)

print("\nTraining samples:", X_train.shape)
print("Validation samples:", X_valid.shape)

# ========== 5. MODEL TRAINING ==========
model = LogisticRegression()
model.fit(X_train, Y_train)

train_score = metrics.roc_auc_score(
    Y_train, model.predict_proba(X_train)[:, 1]
)
valid_score = metrics.roc_auc_score(
    Y_valid, model.predict_proba(X_valid)[:, 1]
)

print("\nTraining ROC-AUC:", train_score)
print("Validation ROC-AUC:", valid_score)

# ========== 6. CONFUSION MATRIX ==========
ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid)
plt.show()

# ========== 7. USER INPUT PREDICTION ==========
print("\n------ STOCK PREDICTION FROM USER INPUT ------")

today_open = float(input("Enter Today's Open Price: "))
today_high = float(input("Enter Today's High Price: "))
today_low = float(input("Enter Today's Low Price: "))
is_quarter_end = int(input("Is it Quarter End? (1 = Yes, 0 = No): "))

open_close = today_open - today_high
low_high = today_low - today_high

user_features = np.array([[open_close, low_high, is_quarter_end]])
user_features = scaler.transform(user_features)

prediction = model.predict(user_features)

if prediction[0] == 1:
    print("📈 Prediction: BUY (Price may go up tomorrow)")
else:
    print("📉 Prediction: DO NOT BUY (Price may go down tomorrow)")

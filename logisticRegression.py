import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_table('insurance_data/ticdata2000.txt')

col_names = [
    "MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR",
    "MGODOV","MGODGE","MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND",
    "MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG","MBERHOOG","MBERZELF",
    "MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2",
    "MSKC","MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS",
    "MZPART","MINKM30","MINK3045","MINK4575","MINK7512","MINK123M",
    "MINKGEM","MKOOPKLA","PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT",
    "PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM","PLEVEN",
    "PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS",
    "PINBOED","PBYSTAND","AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT",
    "AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN",
    "APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL","APLEZIER","AFIETS",
    "AINBOED","ABYSTAND", "CARAVAN"
]

df.columns = col_names

X_small = df.drop('CARAVAN', axis=1) 
y_small = df['CARAVAN']  #targets

y_test_data = np.array(df['CARAVAN'].values)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#aka loss function
def cost_function(X, y, w, b):
    z = np.dot(X, w) + b #prediction, comme a * e + b
    p = sigmoid(z) #or y hat
    cost = -np.mean(y * np.log(p) + (1-y) * np.log(1-p)) #pas obliger de diviser par m vu que : mean
    return cost

def compute_gradients(X, y, w, b, extra_weight=1):
    m = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z) #or y hat

    weights = np.where(y==1, extra_weight, 1) #Ajoute extra_weight si target sinon ajoute rien (x1)

    dw = np.dot(X.T, weights * (p-y)) / np.sum(weights) #transpose pour avoir (n x m) * (m x 1) = n x 1
    db = np.sum(weights * (p-y)) / np.sum(weights) #on divise par weights car c sa la formule (pcq ta besoin du average)
    return dw, db

def logistic_regression(X, y, learning_rate=0.01, iterations=1000, extra_weight=1):
    m, n = X.shape
    w = np.zeros(n) #array of n zeros, on init les weights
    b = 0 #learned bias
    for i in range(iterations):
        cost = cost_function(X, y, w, b)
        dw, db = compute_gradients(X, y, w, b, extra_weight)
        w -= learning_rate*dw
        b -= learning_rate*db
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return w, b

def predict(X, w, b, t):
    z = np.dot(X, w) + b
    p = sigmoid(z)
    #print(np.min(p), np.max(p), np.mean(p))
    return (p >= t).astype(int)


def print_model_stats(X, w, b, threshold):
    y_prediction = predict(X, w, b, threshold)
    accuracy = np.mean(y_prediction == y_test_data)
    conf_matrix = confusion_matrix(y_test_data, y_prediction)
    print(accuracy)
    print(conf_matrix)


# Labels (0 or 1)
# y_test_data = np.array(df['CARAVAN'].values)
# unique, counts = np.unique(y_test_data, return_counts=True)

#w, b = logistic_regression(X_small, y_small, learning_rate=0.01, iterations=1000)
# print("Learned weights:", w)
# print("Learned bias:", b)
# print()
# y_pred = predict(X_small, w, b)
# print("Predictions:", y_pred)
# print("True labels:", y_test_data)
# print()
# print("Confusion Matrix:\n", confusion_matrix(y_small, y_pred))
# print("Accuracy:", accuracy_score(y_small, y_pred))
# print()
# model = LogisticRegression(max_iter=1000)
# model.fit(X_small, y_small)
# y_pred_sklearn = model.predict(X_small)
# print("Sklearn predictions:", y_pred_sklearn)
# print("Sklearn Logistic Regression Accuracy:", accuracy_score(y_small, y_pred_sklearn))


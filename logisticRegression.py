import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from itertools import combinations
import data
import Model

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#aka loss function
def cost_function(X, y, w, b):
    z = np.dot(X, w) + b #prediction, comme a * e + b
    p = sigmoid(z) #or y hat
    cost = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
    return cost

def cost_function_with_l2(X, y, w, b, lambda_const):
    cost = cost_function(X, y, w, b)
    m = X.shape[0]
    l2_reg = (lambda_const / (2*m)) * np.sum(w**2)
    return cost + l2_reg

def compute_gradients(X, y, w, b, extra_weight=1):
    m = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z) #or y hat

    weights = np.where(y==1, extra_weight, 1) #Ajoute extra_weight si target sinon ajoute rien (x1)

    dw = np.dot(X.T, weights * (p-y)) / np.sum(weights) #transpose pour avoir (n x m) * (m x 1) = n x 1
    db = np.sum(weights * (p-y)) / np.sum(weights) #on divise par weights car c sa la formule (pcq ta besoin du average)
    return dw, db

def logistic_regression(X, y, learning_rate=0.01, iterations=1000, extra_weight=1, to_print=True, l2_reg=False, lambda_const=None):
    if l2_reg and lambda_const is None:
        raise ValueError("besoin de lambda_const si l2_reg=True")
    m, n = X.shape
    w = np.zeros(n) #array of n zeros, on init les weights
    b = 0 #learned bias
    for i in range(iterations):
        if l2_reg:
            cost = cost_function_with_l2(X, y, w, b, lambda_const)
        else:
            cost = cost_function(X, y, w, b)
        dw, db = compute_gradients(X, y, w, b, extra_weight)
        if l2_reg:
            dw += (lambda_const / m) * w
        w -= learning_rate*dw
        b -= learning_rate*db
        if to_print and i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return w, b

def predict_probas(X, w, b):
    z = np.dot(X, w) + b
    p = sigmoid(z)
    return p

def predict(X, w, b, t):
    p = predict_probas(X, w, b)
    return (p >= t).astype(int)

def youden_index_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    J = tpr - fpr #TPR == sensitivity, FPR == 1 - specificity
    best_index = np.argmax(J)
    return thresholds[best_index], J[best_index]

def print_model_stats(X, y, w, b, threshold):
    y_prediction = predict(X, w, b, threshold) #Rappel : thresholded -> accuracy et conf matrix
    y_probas = predict_probas(X, w, b)
    accuracy = np.mean(y_prediction == y)
    conf_matrix = confusion_matrix(y, y_prediction)
    auc = roc_auc_score(y, y_probas)
    print('Threshold: ', threshold)
    print('Accuracy: ', accuracy)
    print(conf_matrix)
    print('AUC: ', auc)

def interactions_terms_tester(X_old, y, w_old, b_old, vars_to_test, learning_rate, ratio):
    y_probas = predict_probas(X_old, w_old, b_old)
    auc_base = roc_auc_score(y, y_probas)

    for comb in combinations(vars_to_test, 2):
        X_interaction = X_old.copy()
        inter_name = f"{comb[0]}_x_{comb[1]}"
        X_interaction[inter_name] = X_interaction[comb[0]] * X_interaction[comb[1]]
        X_interaction = X_interaction.to_numpy()
        w,b = logistic_regression(X_interaction, y, learning_rate=learning_rate, extra_weight=ratio, to_print=False)
        y_pred_new = predict_probas(X_interaction, w, b)
        new_auc = roc_auc_score(y, y_pred_new)
        print(f"Interaction {inter_name} : AUC = {new_auc:.4f} (gain = {new_auc - auc_base:+.4f})")

def compare_auc_score(X_old, y, w_old, b_old, X_new, learning_rate, ratio):
    y_probas = predict_probas(X_old, w_old, b_old)
    auc_base = roc_auc_score(y, y_probas)

    w, b = logistic_regression(X_new, y, learning_rate=learning_rate, extra_weight=ratio, to_print=False)
    y_probas_new = predict_probas(X_new, w, b)
    new_auc = roc_auc_score(y, y_probas_new)
    print(f"New X : AUC = {new_auc:.4f} (gain = {new_auc - auc_base:+.4f})")
    return w, b

def find_best_lambda(lambdas, X_train, y_train, val_size=0.2, random_state=42):
    best_lambda = None
    best_auc = 0

    #Ici il faut split le training data pour valider le modèle au lieu d'utiliser le test data
    #car on est entrain de  train le modèle ici, tandis que lorsqu'on comparais les auc score
    # on évaluais le modèle après l'avoir train 
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    for lam in lambdas:
        w, b = logistic_regression(X_train_part, y_train_part, l2_reg=True, lambda_const=lam, to_print=False)
        y_val_probas = predict_probas(X_val, w, b)
        auc = roc_auc_score(y_val, y_val_probas)

        print(f"Lambda: {lam}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_lambda = lam
    print("Best lambda:", best_lambda)
    return best_lambda, best_auc

def overfitting_test(model_old, X_test, model_new, X_test_final):
    y_test_data = data.get_test_targets().to_numpy()
    compare_model_stats(X_test, y_test_data, model_old, model_new, X_new=X_test_final, isTestData=True)

def f1_score_test(w, b):
    X_test = data.get_test_data_with_terms()
    y_test_data = data.get_test_targets()
    #y_probas = predict_probas(X_test, w, b)

    thresholds = np.arange(0,1.01,0.01)
    f1_scores = []

    for t in thresholds:
        y_pred = predict(X_test, w, b, t)
        f1_scores.append(f1_score(y_test_data, y_pred))

    best_thresh = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    print(f"Best threshold est: {best_thresh}\nBest F1 est: {best_f1}")
    return best_thresh

def compare_model_stats(X, y, model_old, model_new, X_new=None, y_new=None, isTestData=False):
    if X_new is None:
        X_new = X
    if y_new is None:
        y_new = y
    if isTestData:
        print("(Avec test data)\n")
    else:
        print("(Avec training data)\n")
    print("Old model :")
    model_old.print_stats(X, y)
    print("\nNew model :")
    model_new.print_stats(X_new, y_new)
    
def create_test_and_X_model(w, b, threshold, interactions=None):
    X_test = data.get_test_data()
    if interactions is not None:
        X_test = data.add_interactions_terms(X_test, interactions)
    X_test_final = X_test.to_numpy()
    test_model = Model.Create(w, b, threshold)
    return test_model, X_test_final
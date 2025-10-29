import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

data = pd.read_csv("task_data.csv")

data["CTR - Cardiothoracic Ratio"] = [ x.replace(",", ".") for x in data["CTR - Cardiothoracic Ratio"]]
data["Inscribed circle radius"] = [ x.replace(",", ".") for x in data["Inscribed circle radius"]]
data["Heart perimeter"] = [ x.replace(",", ".") for x in data["Heart perimeter"]]

X = data[[
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius",
    "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"
]]

y = data["Cardiomegaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

#Decision Tree method

param_grid_tree = {
    "max_depth": range(1, 10),
    "min_samples_leaf": range(1, 10),
    "min_samples_split": range(2, 10),
    'criterion': ['gini', 'entropy'],
}

rskf = RepeatedStratifiedKFold(
    n_splits = 5,
    n_repeats = 100,
    random_state = None
)


grid_search_tree = GridSearchCV (
    estimator = DecisionTreeClassifier(),
    param_grid = param_grid_tree,
    cv = rskf,
    scoring = "accuracy",
    n_jobs = -1,
    error_score = "raise"
)

grid_search_tree.fit(X_train, y_train)

best_params_tree = grid_search_tree.best_estimator_

clf_tree = best_params_tree

clf_tree.fit(X_train, y_train)

cv_score_tree = np.round(cross_val_score(clf_tree, X_train, y_train), 2)

print(f"Scores of cross-validation for each fold using Decision Tree:")
list(map(print, cv_score_tree))
print(f"Cross-validation mean score: {np.mean(cv_score_tree):.4f}")
print(f"Standard deviation of CV score: {np.std(cv_score_tree):.4f}\n")

#KNN method

param_grid_knn = {
    "model__n_neighbors": range(1, 10),
    "model__weights": ["uniform", "distance"],
    "model__metric": ["minkowski", "manhattan", "euclidean", "chebyshev"]
}

rskf = RepeatedStratifiedKFold(
    n_splits = 5,
    n_repeats = 100,
    random_state = None
)


grid_search_knn = GridSearchCV(
    estimator = Pipeline(steps = [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    param_grid = param_grid_knn,
    scoring = "accuracy",
    cv = rskf,
    verbose = 1,
    n_jobs = -1,
    error_score ='raise'
)

grid_search_knn.fit(X_train, y_train)

best_params_knn = grid_search_knn.best_estimator_

pipe_knn = best_params_knn

pipe_knn.fit(X_train, y_train)

cv_score_knn = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

print(f"\nScores of cross-validation for each fold using KNN:")
list(map(print, cv_score_knn))
print(f"Cross-validation mean score: {np.mean(cv_score_knn):.4f}")
print(f"Standard deviation of CV score: {np.std(cv_score_knn):.4f}\n")

#SVM

param_grid_svm = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [0.1, 1, 10, 100, 1000],
    "kernel": ["rbf"]
}

rskf = RepeatedStratifiedKFold(
    n_splits = 5,
    n_repeats = 100,
    random_state = None
)

grid_search_svc = GridSearchCV(
    estimator = SVC(),
    param_grid = param_grid_svm,
    scoring = "accuracy",
    cv = rskf,
    verbose = 1,
    n_jobs = -1,
    error_score = 'raise'
)

grid_search_svc.fit(X_train, y_train)

best_params_svc = grid_search_svc.best_estimator_

pipe_svc = best_params_svc

pipe_svc.fit(X_train, y_train)

cv_score_svc = np.round(cross_val_score(pipe_svc, X_train, y_train), 2)

print(f"\nScores of cross-validation for each fold using SVC:")
list(map(print, cv_score_svc))
print(f"Cross-validation mean score: {np.mean(cv_score_svc):.4f}")
print(f"Standard deviation of CV score: {np.std(cv_score_svc):.4f}\n")



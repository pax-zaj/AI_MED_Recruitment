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

x = data[[
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius",
    "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"
]]

y = data["Cardiomegaly"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_scaled_train = scaler.fit_transform(x_train)
x_scaled_test = scaler.transform(x_test)

#first method - KNN

param_grid_knn = {
    "n_neighbours": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski", "chebyshev"]
}

rsfk = RepeatedStratifiedKFold(
    n_splits = 5,
    n_repeats = 100,
    random_state = None
)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator = pipe_knn,
    param_grid = param_grid_knn,
    scoring = "accuracy",
    cv = rsfk,
    verbose = 1,
    n_jobs = -1
)

grid_search.fit(x_train, y_train)

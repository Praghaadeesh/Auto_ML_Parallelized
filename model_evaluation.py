import ray
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your models
models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    XGBClassifier(random_state=42),
    SVC(kernel='rbf', random_state=42)
]

@ray.remote
def evaluate_model(model, X, y):
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=5)
    end_time = time.time()
    duration = end_time - start_time
    return model.__class__.__name__, scores.mean(), duration

def find_best_model(X, y):
    start_time = time.time()
    ray.init()
    model_evaluations = [evaluate_model.remote(model, X, y) for model in models]
    results = ray.get(model_evaluations)
    ray.shutdown()
    end_time = time.time()
    total_time = end_time - start_time

    best_model_name, best_score, best_duration = max(results, key=lambda x: x[1])
    print(f"Best model: {best_model_name}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best model training time: {best_duration:.2f} seconds")
    print(f"Total time taken (parallel): {total_time:.2f} seconds")

if __name__ == "__main__":
    find_best_model(X_train, y_train)
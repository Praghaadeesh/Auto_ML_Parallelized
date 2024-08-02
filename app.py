import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import ray
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from itertools import product
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import cross_val_score
import random
from Feature_engineering import AutoFeatureEngineering

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

models = {
    'classification': [
        RandomForestClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        XGBClassifier(random_state=42),
        SVC(kernel='rbf', random_state=42)
    ],
    'regression': [
        LinearRegression(),
        RandomForestRegressor(random_state=42),
        XGBRegressor(random_state=42),
        BayesianRidge()
    ]
}

@ray.remote
def train_model(model_cls, params, X, y):
    if type(model_cls) == RandomForestClassifier:
        model = RandomForestClassifier(**params)
    elif type(model_cls) == RandomForestRegressor:
        model = RandomForestRegressor(**params)
    elif type(model_cls) == DecisionTreeClassifier:
        model = DecisionTreeClassifier(**params)
    elif type(model_cls) == XGBClassifier:
        model = XGBClassifier(**params)
    elif type(model_cls) == XGBRegressor:
        model = XGBRegressor(**params)
    elif type(model_cls) == SVC:
        model = SVC(**params)
    elif type(model_cls) == LinearRegression:
        model = LinearRegression()
    elif type(model_cls) == BayesianRidge:
        model = BayesianRidge()
    else:
        raise ValueError(f"Unsupported model class: {type(model_cls)}")

    model.fit(X, y)
    return model


@ray.remote
def cross_validate(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return cv_scores.mean()

@ray.remote
def tune_hyperparameters(model_cls, X, y, cv=5):
    start_time = time.time()
    best_score = -np.inf
    best_model = None
    best_params = None

    if isinstance(model_cls, RandomForestClassifier) or isinstance(model_cls, RandomForestRegressor):
        param_dist = {
            'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=5)],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=3)],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 4],
            'bootstrap': [True, False]
        }
    elif isinstance(model_cls, XGBClassifier) or isinstance(model_cls, XGBRegressor):
        param_dist = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 1000],
            'max_depth': [3, 6, 10],
            'min_child_weight': [1, 5],
            'gamma': [0.0, 0.2],
            'subsample': [0.6, 1.0],
            'colsample_bytree': [0.6, 1.0],
            'reg_alpha': [0, 0.01],
            'reg_lambda': [0.1, 1.0]
        }
    elif isinstance(model_cls, SVC):
        param_dist = {
            'C': [0.1, 10], 
            'kernel': [ 'rbf', 'poly'],  
            'degree': [2, 4],  
            'gamma': ['scale', 'auto'],  
            'coef0': [ 0.1, 0.5, ],  
            'shrinking': [True, False],  
            'probability': [True, False],  
            'tol': [1e-3 ,1e-5],  
            'class_weight': [None, 'balanced']  
        }
    elif isinstance(model_cls, BayesianRidge):
        param_dist = {
            'n_iter': [100, 300],  
            'tol': [1e-3, 1e-4],  
            'alpha_1': [1e-6, 1e-2],  
            'alpha_2': [1e-6, 1e-4],  
            'lambda_1': [ 1e-4, 1e-2],  
            'lambda_2': [1e-6, 1e-2], 
            'alpha_init': [None, 1e-3],  
            'lambda_init': [None, 1e-3],  
            'fit_intercept': [True, False],  
            'normalize': [True, False],  
            'copy_X': [True, False], 
        }
    else:
        param_dist = {}

    
    all_params = [dict(zip(param_dist.keys(), values))
                  for values in itertools.product(*param_dist.values())]

   
    all_params = all_params[:25]
   
    model_evaluations = [train_model.remote(model_cls, params, X, y) for params in all_params]
    model_results = ray.get(model_evaluations)

    # Parallelize cross-validation
    cv_evaluations = [cross_validate.remote(model, X, y, cv=cv) for model in model_results]
    cv_scores = ray.get(cv_evaluations)

    for model, params, score in zip(model_results, all_params, cv_scores):
        #print('model', model, ' score', score, ' params', params)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    end_time = time.time()
    duration = end_time - start_time
    return best_model, best_score, duration, best_params

def find_best_model(X, y, task_type):
    start_time = time.time()
    ray.init()
    model_tuning = [tune_hyperparameters.remote(model_cls, X, y)
                    for model_cls in models[task_type]]
    results = ray.get(model_tuning) 
    ray.shutdown()

    best_model, best_score, best_duration, best_params = max(results, key=lambda x: x[1])
    model_path = create_best_model(best_model, best_params, task_type, X, y)

    names = [model.__class__.__name__ for model in models[task_type]]
    scores = [result[1] for result in results]
    zipped = dict(zip(scores,names))
    sorted_zipped = dict(sorted(zipped.items()))
    print(sorted_zipped)

    # RGB values for colors
    figure_bg_color = (0.5, 0.5, 0.5)
    axes_bg_color = (0.8, 0.8, 0.8)
    text_color = 'white'

    # List of colors for bars
    bar_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

    plt.figure(figsize=(10, 6), facecolor=figure_bg_color)

    # Get the values and keys from the sorted dictionary
    values = list(sorted_zipped.values())
    keys = list(sorted_zipped.keys())

    # Plot the bars with different colors and black outline
    bars = plt.bar(values, keys, color=[bar_colors[i % len(bar_colors)] for i in range(len(keys))], edgecolor='black')

    # Add a black outline to the bars
    for bar in bars:
        bar.set_edgecolor('black')

    plt.xlabel('Models', color=text_color, fontweight='bold')
    plt.ylabel('Accuracy', color=text_color, fontweight='bold')
    plt.title('Model Comparison', color=text_color, fontweight='bold')

    ax = plt.gca()
    ax.set_facecolor(axes_bg_color)
    ax.tick_params(colors=text_color, which='both') 

    llim = max(0, min(scores)-0.05)
    ulim = min(1, max(scores)+0.05)
    plt.ylim(llim, ulim)
    
    image_path = os.path.join(app.root_path, 'static', 'result_graph.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.2, facecolor=figure_bg_color)  # Save the bar graph with tight bounding box
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time

    return best_model.__class__.__name__, best_score, best_duration, total_time, best_params, model_path, image_path

def create_best_model(best_model, best_params, task_type, X, y):
    bmodel = best_model.set_params(**best_params)
    bmodel.fit(X, y)

    model_path = os.path.join(app.root_path, 'Best_models', f'bestmodel_{best_model.__class__.__name__}.pkl')
    pickle.dump(bmodel, open(model_path, 'wb'))
    return model_path

@app.route('/download_model', methods=['POST'])
def download_model():
    model_path = request.form['model_name']
    return send_file(model_path, as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    msg=''
    if request.method == 'POST':
        start_time = time.time()  # Define start_time here
        file = request.files['file']
        target = str(request.form['target'])
        task_type = request.form['task_type']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            df = pd.read_csv(file_path)
            
            if (target not in df.columns):
                msg="Target column doesn't exist, Try again"
            else :
                X = df.drop(target, axis=1)
                y = df[target]
                total_time = time.time() - start_time
                best_model_name, best_score, best_duration, total_time, best_params, model_path, image_path = find_best_model(X, y, task_type)
                os.remove(file_path)
                if(task_type=='classification'):
                    num_models=77
                else:
                    num_models=76

                return render_template('result.html', best_model_name=best_model_name, best_score=best_score,
                                    best_duration=best_duration, total_time=total_time, best_params=best_params,
                                    model_path=model_path, image_path=image_path, num_models=num_models)
    return render_template('index.html', msg=msg)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(os.path.join(app.root_path, 'Best_models')):
        os.makedirs(os.path.join(app.root_path, 'Best_models'))
    app.run(debug=True)
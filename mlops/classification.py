import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import gender_guesser.detector as gender

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter

import os
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def run_classification_pipeline(PROCESSED_DATA_PATH):

    ENCODER_DIR = "./models/encoders"

    users_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "users_features.csv"))
    users_df = users_df[users_df['gender'] != "none"].copy()

    # FEATURE ENGINEERING 
    d = gender.Detector()

    def get_name_gender(name):
        if pd.isna(name):
            return "unknown"

        name = str(name).split()[0]
        g = d.get_gender(name)

        if g == "male":
            return "male"
        elif g == "female":
            return "female"
        elif g == "mostly_male":
            return "mostly_male"
        elif g == "mostly_female":
            return "mostly_female"
        else:
            return "unknown"

    users_df['first_name'] = users_df['name'].str.split().str[0]
    users_df['name_gender'] = users_df['first_name'].apply(get_name_gender)

  
    name_gender_map = joblib.load(os.path.join(ENCODER_DIR, "name_gender_map.pkl"))
    users_df['name_gender'] = users_df['name_gender'].map(name_gender_map)

    # LOAD ENCODERS
    target_encoder = joblib.load(os.path.join(ENCODER_DIR, "gender_target_encoder.pkl"))
    encoders = joblib.load(os.path.join(ENCODER_DIR, "user_encoders.pkl"))
    expected_cols = joblib.load(os.path.join(ENCODER_DIR, "user_columns.pkl"))
    scaler = joblib.load(os.path.join(ENCODER_DIR, "user_scaler.pkl"))

    # TARGET
    y = target_encoder.transform(users_df['gender'])

    # DROP
    drop_cols = ['gender', 'name', 'first_name', 'usercode', 'code', 'company']
    drop_cols = [c for c in drop_cols if c in users_df.columns]

    X = users_df.drop(columns=drop_cols).copy()

    # ENCODING 
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else le.transform([le.classes_[0]])[0]
            )


    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_cols]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SCALE
    X_train_lr = scaler.transform(X_train)
    X_test_lr = scaler.transform(X_test)

    # EVALUATION
    def evaluate(model, X_data, threshold=0.5):

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_data)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            preds = model.predict(X_data)

        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        }

    def log_model(model, name, model_type, X_data, threshold=0.5, extra=None):

        with mlflow.start_run(run_name=f"{name}_{model_type}".lower()):

            metrics = evaluate(model, X_data, threshold)

            mlflow.log_params({
                "model": name,
                "type": model_type,
                "threshold": threshold
            })

            if extra:
                mlflow.log_params(extra)

            mlflow.log_metrics(metrics)

            score = (metrics["accuracy"] + metrics["f1"]) / 2
            mlflow.log_metric("score", score)

            
            if "xgboost" in str(type(model)).lower():

                mlflow.xgboost.log_model(
                    model,
                    name="model"   
                )

            else:
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format="skops"
                )
    mlflow.set_experiment("User Gender Prediction")

    # LOGISTIC REGRESSION (USE SCALED)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_lr, y_train)
    log_model(lr, "Logistic Regression", "Base", X_test_lr)


    # DECISION TREE (BASE)
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    log_model(dt, "Decision Tree", "Base", X_test)


    # DECISION TREE (TUNED)
    params = {
        'max_depth': [3, 5, 8, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        param_grid=params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=1
    )
    grid.fit(X_train, y_train)

    best_dt = grid.best_estimator_

    probs = best_dt.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0

    for t in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    log_model(best_dt, "Decision Tree", "Tuned", X_test, best_thr, grid.best_params_)


    # RANDOM FOREST (BASE)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    log_model(rf, "Random Forest", "Base", X_test)


    # RANDOM FOREST (TUNED)
    params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True],
        'ccp_alpha': [0.0, 0.001, 0.01]
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_distributions=params,
        n_iter=40,
        cv=5,
        scoring='f1',
        n_jobs=1,
        random_state=42
    )
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_

    probs = best_rf.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0

    for t in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    log_model(best_rf, "Random Forest", "Tuned", X_test, best_thr, search.best_params_)


    # XGBOOST (BASE)
    from collections import Counter

    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    xgb.fit(X_train, y_train)
    log_model(xgb, "XGBoost", "Base", X_test)


    # XGBOOST (TUNED)
    params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1],
        'colsample_bytree': [0.7, 0.8, 1],
        'gamma': [0, 0.1, 0.3, 0.5],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5]
    }

    search = RandomizedSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ),
        param_distributions=params,
        n_iter=40,
        cv=5,
        scoring='f1',
        n_jobs=1,
        random_state=42
    )
    search.fit(X_train, y_train)

    best_xgb = search.best_estimator_

    probs = best_xgb.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0

    for t in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    log_model(best_xgb, "XGBoost", "Tuned", X_test, best_thr, search.best_params_)


    print("All classification models logged to MLflow ✅")
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def run_regression_pipeline(PROCESSED_DATA_PATH):

    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "flights_features.csv"))

    # TARGET + DROP
    y = df['log_price']

    drop_cols = ['price','log_price','usercode','travelcode','date', 'from', 'to']
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).copy()

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ENCODING 
    encoders = {}

    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()

        X_train[col] = le.fit_transform(X_train[col].astype(str))

        X_test[col] = X_test[col].astype(str).map(
            lambda x: le.transform([x])[0]
            if x in le.classes_
            else -1  
        )

        encoders[col] = le

    # SAVE ENCODERS 
    os.makedirs("./models/encoders", exist_ok=True)
    joblib.dump(encoders, "./models/encoders/flight_encoders.pkl")
    joblib.dump(X_train.columns.tolist(), "./models/encoders/flight_columns.pkl")

    # SCALING 
    scaler = StandardScaler()

    X_train_base = X_train.copy()
    X_test_base = X_test.copy()

    X_train_lr = scaler.fit_transform(X_train_base)
    X_test_lr = scaler.transform(X_test_base)

    joblib.dump(scaler, "./models/encoders/flight_scaler.pkl")

    # MLflow
    mlflow.set_experiment("Flight Price Prediction")

    def evaluate(model, X_test, X_train):
        y_pred = model.predict(X_test)

        y_pred_real = np.expm1(y_pred)
        y_test_real = np.expm1(y_test)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

        score = (
            (1 / (mae + 1e-6)) * 0.3 +
            (1 / (rmse + 1e-6)) * 0.3 +
            r2 * 0.4
        )

        return mae, rmse, r2, cv, score

    def log_model(model, name, model_type, X_test, X_train, extra=None):

        with mlflow.start_run(run_name=f"{name}_{model_type}".lower()):

            mae, rmse, r2, cv, score = evaluate(model, X_test, X_train)

            mlflow.log_param("model", name)
            mlflow.log_param("type", model_type)

            if extra:
                mlflow.log_params(extra)

            mlflow.log_metrics({
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "CV": cv,
                "score": score
            })

             
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

    # MODELS


    # LINEAR REGRESSION
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train)
    log_model(lr, "Linear Regression", "Base", X_test_lr, X_train_lr)


    # DECISION TREE (BASE)
    dt = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt.fit(X_train_base, y_train)
    log_model(dt, "Decision Tree", "Base", X_test_base, X_train_base)


    # DECISION TREE (TUNED)
    dt_params = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid=dt_params,
        cv=3,
        scoring='r2',
        n_jobs=1
    )

    dt_grid.fit(X_train_base, y_train)

    best_dt = dt_grid.best_estimator_
    log_model(best_dt, "Decision Tree", "Tuned", X_test_base, X_train_base, dt_grid.best_params_)


    # RANDOM FOREST (BASE)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train_base, y_train)
    log_model(rf, "Random Forest", "Base", X_test_base, X_train_base)


    # RANDOM FOREST (TUNED)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=rf_params,
        cv=3,
        scoring='r2',
        n_jobs=1
    )

    rf_grid.fit(X_train_base, y_train)

    best_rf = rf_grid.best_estimator_
    log_model(best_rf, "Random Forest", "Tuned", X_test_base, X_train_base, rf_grid.best_params_)


    # XGBOOST (BASE)
    xgb = XGBRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=1,             
        eval_metric='rmse'
    )

    xgb.fit(X_train_base, y_train)
    log_model(xgb, "XGBoost", "Base", X_test_base, X_train_base)


    # XGBOOST (TUNED)
    xgb_params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }

    xgb_grid = GridSearchCV(
        XGBRegressor(
            random_state=42,
            eval_metric='rmse',
            n_jobs=1  
        ),
        param_grid=xgb_params,
        cv=3,
        scoring='r2',
        n_jobs=1
    )

    xgb_grid.fit(X_train_base, y_train)

    best_xgb = xgb_grid.best_estimator_
    log_model(best_xgb, "XGBoost", "Tuned", X_test_base, X_train_base, xgb_grid.best_params_)


    print("All regression models logged to MLflow ✅")
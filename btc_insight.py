import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split ,TimeSeriesSplit



log_file = open("optuna_output_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file


df=pd.read_csv(r"sample.csv or your dataset")
df_recent = df[df["reserve"].notna()].copy()
df_recent["target"] = (df_recent["close"].shift(-30) > df_recent["close"]).astype(int)
features = [
    "nupl", "sopr", "num_tweets", "avg_sentiment", 
    "miner_btc_outflow", "bitcoin_trend", "DXY", 
    "CPI", "Fed_Rate", "reserve"  
]
df_model = df_recent.dropna(subset=features + ["target"])


from sklearn.model_selection import train_test_split

X = df_model[features]
y = df_model["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False 
)



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    tscv= TimeSeriesSplit(n_splits=5)
    
    f1_scores=[]
    for train_index, valid_index in tscv.split(X_train):
        X_tr,X_val=X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr,y_val=y_train.iloc[train_index], y_train.iloc[valid_index]
        model.fit(X_tr, y_tr)
        y_pred=model.predict(X_val)
        f1=f1_score(y_val, y_pred)
        f1_scores.append(f1)
    return np.mean(f1_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("best trial:")
print('f1-score:', study.best_trial.value)
print('params:', study.best_trial.params)


best_params = study.best_trial.params
tscv= TimeSeriesSplit(n_splits=5)
fold=1
shap_values_all_folds = []
performance=[]

for train_index, test_index in tscv.split(X_train):
    print(f"Fold {fold}")
    X_tr, X_val=X_train.iloc[train_index], X_train.iloc[test_index]
    y_tr, y_val=y_train.iloc[train_index], y_train.iloc[test_index]
    
    model = XGBClassifier(
    **best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    y_pred=model.predict(X_val)
    y_proba=model.predict_proba(X_val)[:, 1]
    acc=accuracy_score(y_val, y_pred)
    auc=roc_auc_score(y_val, y_proba)
    performance.append((acc, auc))
    print(f"Accuracy:{acc:.4f},AUC:{auc:.4f}")
    explainer=shap.Explainer(model,X_tr)
    shap_values=explainer(X_val)
    shap_values_all_folds.append(shap_values)
    fold+=1
all_shap_values=shap_values_all_folds[0]
for sv in shap_values_all_folds[1:]:
    all_shap_values.values=np.vstack([all_shap_values.values,sv.values])
    all_shap_values.data=np.vstack([all_shap_values.data, sv.data])
    all_shap_values.base_values=np.hstack([all_shap_values.base_values, sv.base_values])
shap.plots.beeswarm(all_shap_values, max_display=10)
accs, aucs = zip(*performance)
print(f"Average Accuracy:{np.mean(accs):.4f}, Average ROC-AUC:{np.mean(aucs):.4f}")  , 

best_model= XGBClassifier(
    **best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(X_train, y_train)
y_pred =best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



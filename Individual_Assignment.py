import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# step 1：数据加载
file = 'Assignment2_mimic dataset.csv'
df = pd.read_csv(file)
print("-> Data load successful.")
print(f"Raw data shape: {df.shape}")

# step 2：数据清洗，防泄露
print("\nCleaning data...")

drop_cols = [
    'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime',
    'deathtime', 'los', 'icu_los_hours', 'last_careunit',
    'hospital_expire_flag', 'radiology_note_text',
    'radiology_note_time_min', 'radiology_note_time_max'
]

df_clean = df.drop(columns=drop_cols, errors='ignore')
X = df_clean.drop(columns=['icu_death_flag'])
y = df_clean['icu_death_flag']
X = X.dropna(axis=1, how='all')
all_cat = X.select_dtypes(include=['object', 'category']).columns
cat_cols = [c for c in all_cat if X[c].nunique() < 15]
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X = X[list(num_cols) + list(cat_cols)]

# 补缺失值
num_imp = SimpleImputer(strategy='median')
cat_imp = SimpleImputer(strategy='most_frequent')

X_num = pd.DataFrame(num_imp.fit_transform(X[num_cols]), columns=num_cols)
X_cat = pd.DataFrame(cat_imp.fit_transform(X[cat_cols]), columns=cat_cols)
X_cat = pd.get_dummies(X_cat, drop_first=True)
X_final = pd.concat([X_num, X_cat], axis=1)

# step 3：特征选择
print("\nTraining feature selector...")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_train, y_train)

selector = SelectFromModel(rf_sel, prefit=True)

X_train_fs = selector.transform(X_train)
X_test_fs = selector.transform(X_test)

sel_feats = X_train.columns[selector.get_support()]

print("\nFeature Selection Results:")
print(f"Features before reduction: {X_final.shape[1]}")
print(f"Features after reduction: {len(sel_feats)}")

print("\nTop 15 features:")

imps = rf_sel.feature_importances_

imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': imps
})

top_feats = imp_df.sort_values(
    by='Importance', ascending=False
).head(15)

print(top_feats.to_string(index=False))

# step 4：预测模型
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_fs, y_train)
lr_prob = lr.predict_proba(X_test_fs)[:, 1]
lr_pred = lr.predict(X_test_fs)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
dt.fit(X_train_fs, y_train)
dt_prob = dt.predict_proba(X_test_fs)[:, 1]
dt_pred = dt.predict(X_test_fs)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_fs, y_train)
rf_prob = rf.predict_proba(X_test_fs)[:, 1]
rf_pred = rf.predict(X_test_fs)

print("Training AdaBoost...")
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train_fs, y_train)
ada_prob = ada.predict_proba(X_test_fs)[:, 1]
ada_pred = ada.predict(X_test_fs)

# evaluation
def eval_model(name, y_true, y_pred, y_prob):
    print(f"\n[{name}]")
    print(f"AUROC     : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_true, y_pred):.4f}")

print("\nModel Comparison Results:")

eval_model("Logistic Regression", y_test, lr_pred, lr_prob)
eval_model("Decision Tree", y_test, dt_pred, dt_prob)
eval_model("Random Forest", y_test, rf_pred, rf_prob)
eval_model("AdaBoost", y_test, ada_pred, ada_prob)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

cat_indices = [2, 3, 4, 5, 6, 10, 13, 15, 16]
ord_indices = [1]
num_indices = [0, 7, 8, 9, 11, 12, 14, 17, 18]
cat_cols = [X_train.columns[i] for i in cat_indices]
ord_cols = [X_train.columns[i] for i in ord_indices]
num_cols = [X_train.columns[i] for i in num_indices]

preprocessor = ColumnTransformer([
    ('one_col', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), cat_cols),
    ('ordi_col', OrdinalEncoder(categories=[['Standard', 'Basic', 'Premium']]), ord_cols),
    ('scaler', StandardScaler(), num_cols)
], remainder='passthrough')

smote = SMOTE(random_state=42)
feature_selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='mean'
)
lrf = LogisticRegression(penalty='l2', C=0.48343714531846416, max_iter=639)

pipe = ImbPipeline([
    ('preprocessing', preprocessor),
    ('smote', smote),
    ('feature_selection', feature_selector),
    ('lrf', lrf)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]

print('accuracy=', round(accuracy_score(y_test, y_pred), 4))
print('precision=', round(precision_score(y_test, y_pred, zero_division=0), 4))
print('recall=', round(recall_score(y_test, y_pred, zero_division=0), 4))
print('f1=', round(f1_score(y_test, y_pred, zero_division=0), 4))
print('roc_auc=', round(roc_auc_score(y_test, prob), 4))
print('confusion_matrix=')
print(confusion_matrix(y_test, y_pred))

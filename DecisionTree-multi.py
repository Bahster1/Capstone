import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample


SEED = 0

df = pd.read_csv('./predictive_maintenance.csv')
df = df.drop(columns=['Product ID', 'UDI', 'Target'])

majority = df[df['Failure Type'] == 'No Failure']
minority = df[df['Failure Type'] != 'No Failure']
majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=SEED)
df = pd.concat([majority_down, minority])

X = pd.get_dummies(df.drop(columns='Failure Type'))
y = df['Failure Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

params = {
	'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier()
grid = GridSearchCV(param_grid=params, estimator=dt, verbose=1)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print('DECISION TREE - MULTI CLASS')
print(classification_report(y_test, y_pred))
print('\nCONFUSION MATRIX')
print(confusion_matrix(y_test, y_pred))

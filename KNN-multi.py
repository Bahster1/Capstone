import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
	'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
	'weights': ['uniform', 'distance'],
	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid = GridSearchCV(param_grid=params, estimator=knn, verbose=1)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print('K-NEAREST NEIGHBORS - MULTI CLASS')
print(classification_report(y_test, y_pred))
print('\nCONFUSION MATRIX')
print(confusion_matrix(y_test, y_pred))

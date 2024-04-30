# https://www.kaggle.com/code/joshuaswords/time-series-anomaly-detection
# https://www.kaggle.com/datasets/thedevastator/improving-naval-vessel-condition-through-machine

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def run_isolation_forest(data: pd.DataFrame, contamination=0.005, n_estimators=200, max_samples=0.7):
	model = IsolationForest(random_state=0, contamination=contamination, n_estimators=n_estimators, max_samples=max_samples)
	model.fit(data)

	outliers = pd.Series(model.predict(data)).apply(lambda x: 1 if x == -1 else 0)
	score = model.decision_function(data)

	return outliers, score


df = pd.read_csv('./data.csv')
df = df.drop(columns=['index'])
df = df.dropna()

outliers, score = run_isolation_forest(df)

df = (df
      .assign(Outliers=outliers)
      .assign(Score=score))

for col in df[:-2]:
	df[col][:100].plot()

	for index, row in df[:100].iterrows():
		if row['Outliers'] == 1:
			plt.plot(index, df[col][index], 'ro')

plt.show()

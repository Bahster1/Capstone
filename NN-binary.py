import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import keras


SEED = 0
EPOCHS = 50

df = pd.read_csv('./predictive_maintenance.csv')
df = df.drop(columns=['Product ID', 'UDI', 'Failure Type'])

majority = df[df['Target'] == 0]
minority = df[df['Target'] == 1]
majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=SEED)
df = pd.concat([majority_down, minority])

X = pd.get_dummies(df.drop(columns='Target'))
y = pd.get_dummies(df['Target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

model = keras.models.Sequential()
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_split=0.2)

print('\nBINARY CLASSIFICATION MODEL')
model.evaluate(X_test, y_test)

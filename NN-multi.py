import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import keras


SEED = 0
EPOCHS = 50

df = pd.read_csv('./predictive_maintenance.csv')
df = df.drop(columns=['Product ID', 'UDI', 'Target'])

majority = df[df['Failure Type'] == 'No Failure']
minority = df[df['Failure Type'] != 'No Failure']
majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=SEED)
df = pd.concat([majority_down, minority])

X = pd.get_dummies(df.drop(columns='Failure Type'))
y = df['Failure Type']

le = LabelEncoder()
le.fit(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
y_train = pd.get_dummies(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = le.inverse_transform(y_pred)

print('FEED-FORWARD NEURAL NETWORK - MULTI CLASS')
print(classification_report(y_test, y_pred))
print('\nCONFUSION MATRIX')
print(confusion_matrix(y_test, y_pred))

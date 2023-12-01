import tensorflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
LE = LabelEncoder()

"""
print(df.isnull().sum())
print(f'\nNumber of duplicates: {df.duplicated().sum()}\n')
print(df.dtypes)
"""
# NO DUPLICATE, NULL, OR EMPTY DATA

# Converting categorical data to numerical
df['Gender'] = LE.fit_transform(df['Gender'])
df['Occupation'] = LE.fit_transform(df['Occupation'])
df['BMI Category'] = LE.fit_transform(df['BMI Category'])
df['Blood Pressure'] = LE.fit_transform(df['Blood Pressure'])
df['Sleep Disorder'] = LE.fit_transform(df['Sleep Disorder'])

# Preprocessing
MMS = MinMaxScaler()  # Used for normalization
SS = StandardScaler()  # Used for Standardization

df['Age'] = SS.fit_transform(df[['Age']])
df['Sleep Duration'] = SS.fit_transform(df[['Sleep Duration']])
df['Quality of Sleep'] = MMS.fit_transform(df[['Quality of Sleep']])
df['Physical Activity Level'] = MMS.fit_transform(df[['Physical Activity Level']])
df['Stress Level'] = MMS.fit_transform(df[['Stress Level']])
df['Blood Pressure'] = SS.fit_transform(df[['Blood Pressure']])
df['Heart Rate'] = SS.fit_transform(df[['Heart Rate']])
df['Daily Steps'] = SS.fit_transform(df[['Daily Steps']])

# Remove the Person ID column
df.drop(['Person ID'], axis=1, inplace=True)

# Extracting X data and Y data
X_data = df[df.columns.drop(['Sleep Disorder'])].values
Y_data = df['Sleep Disorder'].values

# Split the data into 80% training data and 20% test data
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)

# Only apply SMOTE to the training data
sm = SMOTE(random_state=0)
X_train_res, y_train = sm.fit_resample(X_train_temp, y_train_temp.ravel())

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_res)
X_test_scaled = sc.transform(X_test)

X_train_temp, X_test, y_train_temp, y_test_dev = train_test_split(X_data, Y_data, test_size=0.15, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train_temp, y_train_temp.ravel())
X_train, X_valid, y_train_dev, y_valid_dev = train_test_split(X_train_res, y_train_res, test_size=(.15 / .85),
                                                              random_state=0)
sc = StandardScaler()
X_train_scaled_dev = sc.fit_transform(X_train)
X_valid_scaled_dev = sc.transform(X_valid)
X_test_scaled_dev = sc.transform(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled_dev.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled_dev, y_train_dev, epochs=20, batch_size=32, validation_data=(X_valid_scaled_dev, y_valid_dev))
history_dict = history.history

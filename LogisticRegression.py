import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

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

'''
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of logistic regression on the test data is: {round(accuracy, 4)}')
'''

############

X_train_temp, X_test, y_train_temp, y_test_dev = train_test_split(X_data, Y_data, test_size=0.15, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train_temp, y_train_temp.ravel())
X_train, X_valid, y_train_dev, y_valid_dev = train_test_split(X_train_res, y_train_res, test_size=(.15 / .85),
                                                              random_state=0)
sc = StandardScaler()
X_train_scaled_dev = sc.fit_transform(X_train)
X_valid_scaled_dev = sc.transform(X_valid)
X_test_scaled_dev = sc.transform(X_test)


def logistic_regression(x_train, y_train1, x_valid, y_valid, x_test, y_test1):
    logreg_default = LogisticRegression(random_state=0)
    logreg_default.fit(x_train, y_train1)
    y_pred_valid_default = logreg_default.predict(x_valid)

    accuracy_default = accuracy_score(y_valid, y_pred_valid_default)
    print(f'Accuracy score of logistic regression using default hyperparameters on the validation set: {accuracy_default}')

    c_s = [0.0001, 0.001, 0.01, 0.1, 10, 100, 1000, 10000]
    c_improve_accuracy = 0
    c_improve = 0

    for c in c_s:
        logreg_temp = LogisticRegression(C=c)
        logreg_temp.fit(x_train, y_train1)
        y_pred_temp = logreg_temp.predict(x_valid)
        accuracy_temp = accuracy_score(y_valid, y_pred_temp)
        if accuracy_temp > c_improve_accuracy:
            c_improve_accuracy = accuracy_temp
            c_improve = c

    logreg_improve = LogisticRegression(C=c_improve, max_iter=500)
    logreg_improve.fit(x_train, y_train1)
    y_pred_valid_improve = logreg_improve.predict(x_valid)
    accuracy_improve = accuracy_score(y_valid, y_pred_valid_improve)
    print(f'Accuracy score of logistic regression on the validation set with C = {c_improve} is {accuracy_improve}')

    x_train_combined = np.concatenate((x_train, x_valid))
    y_train_combined = np.concatenate((y_train1, y_valid))
    logreg = LogisticRegression(C=c_improve, max_iter=500)
    logreg.fit(x_train_combined, y_train_combined)
    y_pred = logreg.predict(x_test)
    accuracy = accuracy_score(y_test1, y_pred)
    print(f'Accuracy score of logistic regression on the test set with C = {c_improve} is {accuracy}')

# Note for the report on why the accuracy is lower --> see kayla's personal testing file


# Test for calling the function
if __name__ == '__main__':
    logistic_regression(X_train_scaled_dev, y_train_dev, X_valid_scaled_dev, y_valid_dev, X_test_scaled_dev, y_test_dev)
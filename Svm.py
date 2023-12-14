import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
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

X_train_temp, X_test, y_train_temp, y_test_dev = train_test_split(X_data, Y_data, test_size=0.15, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train_temp, y_train_temp.ravel())
X_train, X_valid, y_train_dev, y_valid_dev = train_test_split(X_train_res, y_train_res, test_size=(.15 / .85),
                                                              random_state=0)
sc = StandardScaler()
X_train_scaled_dev = sc.fit_transform(X_train)
X_valid_scaled_dev = sc.transform(X_valid)
X_test_scaled_dev = sc.transform(X_test)



def train_test_SVM(X_train, y_train, X_valid, y_valid, X_test, y_test):
    def get_best_c(X_train_scaled_dev, y_train_dev, X_valid_scaled_dev, y_valid_dev):
        c_values = [0.0001, 0.001, 0.01, 0.1, 10, 100, 1000, 10000]
        best_accuracy = 0
        best_c = 0

        for c in c_values:
            svm_temp = SVC(C=c, kernel='rbf', random_state=0)
            svm_temp.fit(X_train_scaled_dev, y_train_dev)
            y_pred_temp = svm_temp.predict(X_valid_scaled_dev)
            accuracy_temp = accuracy_score(y_valid_dev, y_pred_temp)
            if accuracy_temp > best_accuracy:
                best_accuracy = accuracy_temp
                best_c = c

        return best_c

    svm_default = SVC(kernel='rbf', random_state=0)
    svm_default.fit(X_train, y_train)
    y_pred_valid_default = svm_default.predict(X_valid)
    accuracy_default = accuracy_score(y_valid, y_pred_valid_default)
    print(f'Accuracy score of SVM using default hyperparameters on the validation set: {accuracy_default}')

    best_c = get_best_c(X_train, y_train, X_valid, y_valid)

    svm_final = SVC(C=best_c, kernel='rbf', random_state=0)
    svm_final.fit(np.concatenate((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    y_pred = svm_final.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy score of SVM on the test set with C = {best_c} is {accuracy}')

train_test_SVM(X_train_scaled_dev, y_train_dev, X_valid_scaled_dev, y_valid_dev, X_test_scaled_dev, y_test_dev)




"""



# SVM model
svm_default = SVC(kernel='rbf', random_state=0)
svm_default.fit(X_train_scaled_dev, y_train_dev)
y_pred_valid_default = svm_default.predict(X_valid_scaled_dev)

accuracy_default = accuracy_score(y_valid_dev, y_pred_valid_default)
print(f'Accuracy score of SVM using default hyperparameters on the validation set: {accuracy_default}')

# Hyperparameter tuning for SVM
c_values = [0.0001, 0.001, 0.01, 0.1, 10, 100, 1000, 10000]
best_accuracy = 0
best_c = 0

for c in c_values:
    svm_temp = SVC(C=c, kernel='rbf', random_state=0)
    svm_temp.fit(X_train_scaled_dev, y_train_dev)
    y_pred_temp = svm_temp.predict(X_valid_scaled_dev)
    accuracy_temp = accuracy_score(y_valid_dev, y_pred_temp)
    if accuracy_temp > best_accuracy:
        best_accuracy = accuracy_temp
        best_c = c

svm_improve = SVC(C=best_c, kernel='rbf', random_state=0)
svm_improve.fit(X_train_scaled_dev, y_train_dev)
y_pred_valid_improve = svm_improve.predict(X_valid_scaled_dev)
accuracy_improve = accuracy_score(y_valid_dev, y_pred_valid_improve)
print(f'Accuracy score of SVM on the validation set with C = {best_c} is {accuracy_improve}')

# Training the final SVM model on combined training and validation data
X_train_combined = np.concatenate((X_train_scaled_dev, X_valid_scaled_dev))
y_train_combined = np.concatenate((y_train_dev, y_valid_dev))
svm_final = SVC(C=best_c, kernel='rbf', random_state=0)
svm_final.fit(X_train_combined, y_train_combined)
y_pred = svm_final.predict(X_test_scaled_dev)
accuracy = accuracy_score(y_test_dev, y_pred)
print(f'Accuracy score of SVM on the test set with C = {best_c} is {accuracy}')


"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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


# Random forest
rf_model_initial = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=5,
                                          random_state=0)
rf_model_initial.fit(X_train_scaled_dev, y_train_dev)
y_pred_initial = rf_model_initial.predict(X_test_scaled_dev)

# Initial performance evaluation
print('Accuracy score of default RF: {0:0.4}'.format(accuracy_score(y_test_dev, y_pred_initial)))

# Updating forest
rf_model_temp = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=5,
                                       random_state=0)
rf_model_temp.fit(X_train_scaled_dev, y_train_dev)
y_pred_temp = rf_model_temp.predict(X_valid_scaled_dev)
print('Accuracy score of temp RF: {0:0.4}'.format(accuracy_score(y_valid_dev, y_pred_temp)))

# Tuning hyperparameters
param_grid = {'n_estimators': [15, 25, 50, 75, 100, 125, 150],
              'max_features': ['sqrt', 'log2'],
              'max_depth': [3, 5, 7, 9],
              'min_samples_split': [10, 30, 50, 75]}

rand_search = RandomizedSearchCV(rf_model_temp, param_distributions=param_grid, cv=10, random_state=0)
rand_search.fit(X_train_scaled_dev, y_train_dev)
print(f'Best model:\n {rand_search.best_estimator_}')

# Determining feature importance
importances = rf_model_initial.feature_importances_
feature_names = df.columns.drop(['Sleep Disorder'])

# Creating a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sorting the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Feature importance
top_n = 11
top_features = feature_importance_df.head(top_n)
print(top_features)

# Final model
rf_model_final = RandomForestClassifier(n_estimators=50, max_depth=9, max_features='log2', min_samples_split=10,
                                        random_state=0)
rf_model_final.fit(X_train_scaled_dev, y_train_dev)
y_pred_final = rf_model_final.predict(X_test_scaled_dev)
print('Accuracy score of final RF: {0:0.4}'.format(accuracy_score(y_test_dev, y_pred_final)))

# Confusion Matrix
cm = confusion_matrix(y_test_dev, y_pred_final)
print('Confusion matrix\n\n', cm)

# Classification report
print(f'Classification report of default RF:\n{classification_report(y_test_dev, y_pred_final)}')

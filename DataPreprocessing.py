import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

    print(df.isnull().sum())
    print(f'\nNumber of duplicates: {df.duplicated().sum()}\n')
    print(df.dtypes)

    # Change categorical variables into numeric ones
    # Make dummy variables instead of label encoder in one column -> so one label doesn't have more weight than another
    df['Male'] = 0
    df['Female'] = 0

    df['Doctor'] = 0
    df['Nurse'] = 0
    df['Teacher'] = 0
    df['Software Engineer'] = 0
    df['Sales Representative'] = 0
    df['Engineer'] = 0
    df['Accountant'] = 0
    df['Scientist'] = 0
    df['Lawyer'] = 0
    df['Salesperson'] = 0
    df['Manager'] = 0

    df['Normal'] = 0
    df['Normal Weight'] = 0
    df['Obese'] = 0
    df['Overweight'] = 0

    for idx in range(len(df)):
        if df['Gender'][idx] == 'Male':
            df['Male'][idx] = 1
        elif df['Gender'][idx] == 'Female':
            df['Female'][idx] = 1

        if df['Occupation'][idx] == 'Doctor':
            df['Doctor'][idx] = 1
        elif df['Occupation'][idx] == 'Nurse':
            df['Nurse'][idx] = 1
        elif df['Occupation'][idx] == 'Teacher':
            df['Teacher'][idx] = 1
        elif df['Occupation'][idx] == 'Software Engineer':
            df['Software Engineer'][idx] = 1
        elif df['Occupation'][idx] == 'Sales Representative':
            df['Sales Representative'][idx] = 1
        elif df['Occupation'][idx] == 'Engineer':
            df['Engineer'][idx] = 1
        elif df['Occupation'][idx] == 'Accountant':
            df['Accountant'][idx] = 1
        elif df['Occupation'][idx] == 'Scientist':
            df['Scientist'][idx] = 1
        elif df['Occupation'][idx] == 'Lawyer':
            df['Lawyer'][idx] = 1
        elif df['Occupation'][idx] == 'Salesperson':
            df['Salesperson'][idx] = 1
        elif df['Occupation'][idx] == 'Manager':
            df['Manager'][idx] = 1

        if df['BMI Category'][idx] == 'Normal':
            df['Normal'][idx] = 1
        elif df['BMI Category'][idx] == 'Normal Weight':
            df['Normal Weight'][idx] = 1
        elif df['BMI Category'][idx] == 'Obese':
            df['Obese'][idx] = 1
        elif df['BMI Category'][idx] == 'Overweight':
            df['Overweight'][idx] = 1

    column_move = df.pop('Sleep Disorder')
    df.insert(29, 'Sleep Disorder', column_move)

    # X - features to predict sleep disorder
    X = df.iloc[:, :-1].values
    # y = sleep disorder
    y = df.iloc[:, -1].values

    # Map None, Sleep Apnea, and Insomnia to different values

    df.drop(['Gender'], axis=1, inplace=True)
    df.drop(['Occupation'], axis=1, inplace=True)
    df.drop(['BMI Category'], axis=1, inplace=True)

    print(df.min())
    print(df.max())
    print()
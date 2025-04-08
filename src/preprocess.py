# src/preprocess.py

import os
import pandas as pd
import boto3

def download_from_s3(bucket_name, base_key, local_dir, files):
    s3 = boto3.client('s3')

    os.makedirs(local_dir, exist_ok=True)

    for file in files:
        s3_key = f"{base_key}{file}"
        local_path = os.path.join(local_dir, file)

        s3.download_file(bucket_name, s3_key, local_path)
        print(f"✅ Downloaded {file} to {local_path}")


def load_data(local_dir):
    train = pd.read_csv(f'{local_dir}/train.csv', parse_dates=True, low_memory=False, index_col='Date')
    test = pd.read_csv(f'{local_dir}/test.csv', parse_dates=True, low_memory=False, index_col='Date')
    store = pd.read_csv(f'{local_dir}/store.csv')

    return train, test, store


def preprocess_data(train, test, store):
    train['SalesPerCustomer'] = train['Sales'] / train['Customers']
    store.fillna(0, inplace=True)

    train['Year'] = train.index.year
    train.reset_index(inplace=True)

    test['Year'] = test.index.year
    test.reset_index(inplace=True)

    train_store = train.copy()
    test_store = test.copy()

    train_store['Date'] = pd.to_datetime(train_store['Date'])
    train_store.set_index('Date', inplace=True)

    test_store['Date'] = pd.to_datetime(test_store['Date'])
    test_store.set_index('Date', inplace=True)

    train_store = train_store.resample('D').mean(numeric_only=True).interpolate()
    test_store = test_store.resample('D').mean(numeric_only=True).interpolate()

    store_avg_spc = train_store.groupby('Store')['SalesPerCustomer'].mean()

    train_store['Store'] = train_store['Store'].astype(int)
    test_store['Store'] = test_store['Store'].astype(int)

    test_store['SalesPerCustomer'] = test_store['Store'].map(store_avg_spc)
    test_store['SalesPerCustomer'].fillna(store_avg_spc.mean(), inplace=True)

    test_store['Id'] = test_store['Id'].astype(int)
    test_store['Store'] = test_store['Store'].astype(int)
    train_store['Store'] = train_store['Store'].astype(int)

    train_store['Open'] = (train_store['Open'] > 0.5).astype(int)
    test_store['Open'] = (test_store['Open'] > 0.5).astype(int)

    features = [
        'Store', 'DayOfWeek', 'Open', 'Promo',
        'SchoolHoliday', 'SalesPerCustomer', 'Year'
    ]

    X_train = train_store[features]
    y_train = train_store['Sales']
    X_test = test_store[features]

    return X_train, y_train, X_test, test_store

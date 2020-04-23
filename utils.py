import os
import pandas as pd


class Data():
    def __init__(self, train_val, test):
        self.train_val_df = train_val
        self.test_df = test

def read_data(normalize=True, keep_nan = False, keep_dates=True):
    train_dfs=[]
    test_dfs=[]
    count_per_station = 0
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            print('Loaded ',os.path.join(dirname, filename))
            df = pd.read_csv(os.path.join(dirname, filename))

            # 70:15:15 split ratio for Train, Val, Test
            count_per_station = df.shape[0]
            test_data_count = int(count_per_station*0.15)
            train_data_count = count_per_station - test_data_count 

            train_df = df[0: train_data_count]
            test_df = df[train_data_count :]
            
            train_dfs.append(train_df)
            test_dfs.append(test_df)
    
    df_all_train = pd.concat(train_dfs,ignore_index=True)
    df_all_test = pd.concat(test_dfs,ignore_index=True)
    
    # For normalization use
    all_dfs = pd.concat([df_all_train,df_all_test],ignore_index=True)

    def transform_df(d):
        # Convert year month day to numerical values unit in hour
        if keep_dates:
            d['time_stamp'] = pd.to_datetime(d[["year", "month", "day"]])
            d = d.drop(columns=['No'])
        else:
            d = d.rename(columns={"No": "time_stamp"})
        # Drop NA rows
        if not keep_nan:
            d = d.dropna()
        ## One hot encoding for wind direction
        dfDummies = pd.get_dummies(d['wd'], prefix = 'WD')
        if normalize:
            cols_to_norm = ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
            for feature_name in cols_to_norm:
                max_value = all_dfs[feature_name].max()
                min_value = all_dfs[feature_name].min()
                d[feature_name] = (d[feature_name] - min_value) / (max_value - min_value)
            print("Applied normalization on ",cols_to_norm)
        d = pd.concat([d, dfDummies], axis=1)
        # Drop useless cols
        d = d.drop(columns=['year', 'month', 'day', 'hour', 'wd'])
        return d

    df_all_train = transform_df(df_all_train)
    df_all_test = transform_df(df_all_test)
        
    return Data(df_all_train,df_all_test)


def prediction_accuracy(predict_value, true_value, tolerance):
    diff = predict_value - true_value
    return float(len(diff[abs(diff)<=tolerance])) / len(diff)


if __name__ == '__main__':
    dataset = read_data()
    print(dataset.train_val_df.describe())
    print(dataset.test_df.describe())
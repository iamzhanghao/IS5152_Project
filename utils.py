import os
import pandas as pd

def read_data(normalize=True):
    dfs=[]
    count_per_station = 0
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            print('Loaded ',os.path.join(dirname, filename))
            df = pd.read_csv(os.path.join(dirname, filename))
            count_per_station = df.shape[0]
            dfs.append(df)
    df_all = pd.concat(dfs,ignore_index=True)
    # Convert year month day to numerical values unit in hour
    #df_all = df_all.rename(columns={"No": "time_stamp"})
    df_all['time_stamp'] = pd.to_datetime(df_all[["year", "month", "day"]])
    # Drop NA rows
    df_all = df_all.dropna()
    ## One hot encoding for wind direction
    dfDummies = pd.get_dummies(df_all['wd'], prefix = 'WD')
    if normalize:
        cols_to_norm = ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
        print("Applied normalization on ",cols_to_norm )
        df_all[cols_to_norm] = df_all[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_all = pd.concat([df_all, dfDummies], axis=1)
    # Drop useless cols
    df_all = df_all.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd'])
    print("%d rows per station, total %d rows" % (count_per_station, df_all.shape[0]))
    return count_per_station, df_all

def split_data(df):
    # split data into trainng 90% and test 10%
    # split training data into training set and validation set 80:20
    total_data_count = len(df)
    test_data_count = int(total_data_count*0.1)
    train_data_count = int((total_data_count - test_data_count)*0.8)
    validation_data_count = total_data_count - test_data_count - train_data_count

    train_df = df[0: train_data_count]
    validation_df = df[train_data_count : train_data_count + validation_data_count]
    test_df = df[train_data_count + validation_data_count :]
    print("Data is split into train, validation and test dataset successfully!")
    return train_df, validation_df, test_df
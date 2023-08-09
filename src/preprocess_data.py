import os
import sys
import pandas as pd
from sklearn import preprocessing


def count_nulls_by_line(df):
    return df.isnull().sum().sort_values(ascending=False)


def null_percent_by_line(df):
    return (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)


def preprocess_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Drop columns with missing values
    dropList = list(df.columns[df.isnull().mean() > 0.15])
    df.drop(dropList, axis=1, inplace=True)
    df.drop(["Date", "Location"], axis=1, inplace=True)

    # Convert RainToday and RainTomorrow to binary labels
    df["RainToday"] = df["RainToday"].astype(str)
    df["RainTomorrow"] = df["RainTomorrow"].astype(str)
    lb = preprocessing.LabelBinarizer()
    df["RainToday"] = lb.fit_transform(df["RainToday"])
    df["RainTomorrow"] = lb.fit_transform(df["RainTomorrow"])

    # Perform one-hot encoding on selected categorical columns
    categorical_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    df = pd.get_dummies(data=df, columns=categorical_columns)

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Reorder the columns
    cols = df.columns.tolist()
    cols.remove("RainTomorrow")
    cols.append("RainTomorrow")
    df = df[cols]

    # Save processed data to CSV files
    features_df = df.drop(["RainTomorrow"], axis=1)
    features_df.to_csv("./data/features.csv", index=False)
    df.to_csv(DATA_PATH[:-4] + "_processed.csv", index=False)


if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])
    preprocess_data(DATA_PATH)
    print("Saved to {}".format(DATA_PATH[:-4] + "_processed.csv"))

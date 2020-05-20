import pandas as pd
import numpy as np
from os.path import join

from sklearn import datasets as sklearn_datasets
from sklearn.preprocessing import StandardScaler

def prepare_adult(train_data, test_data, binarize=False):
    if type(train_data) == str:
        train_data = pd.read_csv(train_data)
        test_data = pd.read_csv(test_data)

    df = pd.concat([train_data, test_data], ignore_index=True)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    num_cols = ["hours-per-week", "capital-loss", "capital-gain"]

    cat_cols = [
        "work-class",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    if binarize:
        cat_cols.append("education")
        num_cols.remove("capital-loss")
        num_cols.remove("capital-gain")
    else:
        num_cols.append("education-num")
        scaler = StandardScaler().fit(df[num_cols])
        df[num_cols] = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)

    # one_hot_encoded = pd.get_dummies(df[cat_cols])
    #
    # quant_encoded_parts = []
    # for quant_col in quant_cols:
    #     try:
    #         bins = pd.qcut(df[quant_col], quantiles)
    #
    #     except ValueError:
    #         bins = pd.qcut(df[quant_col], quantiles, duplicates="drop")
    #         num_bins = len(bins.unique())
    #         warnings.warn(
    #             "%s will have fewer bins than specified: %d" % (quant_col, num_bins)
    #         )
    #
    #     quant_encoded_parts.append(pd.get_dummies(bins, prefix=quant_col))
    #
    # quant_encoded_parts.append(
    #     pd.get_dummies(pd.qcut(df[target_col], num_classes), prefix=target_col)
    # )
    #
    # parts = [one_hot_encoded] + quant_encoded_parts
    # if not binarize:a
    #     parts.append(scaled)

    return df[cat_cols + num_cols + ["income"]]

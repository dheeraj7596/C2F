import sys
import json
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    data_dir = sys.argv[1]
    with open(os.path.join(data_dir, "parent_to_child.json")) as f:
        parent_to_child = json.load(f)

    df_list = []
    for p in parent_to_child:
        temp_df = pickle.load(open(os.path.join(data_dir, "preds_" + p + ".pkl"), "rb"))
        df_list.append(temp_df)

    df_final = pd.concat(df_list).reset_index(drop=True)
    print(classification_report(df_final["label"], df_final["pred"]))

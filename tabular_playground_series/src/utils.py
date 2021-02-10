import numpy as np
import pandas as pd

def categorical_features_encoding(df):
    df_copy = df.copy()

    label_enc_fet = ['cat'+str(i) for i in range(3)]
    onehot_enc_fet = ['cat'+str(i) for i in range(3, 10)]

    mapping = {'A':0, 'B':1}
    for col in label_enc_fet:
        df_copy[col] = df_copy[col].map(mapping)

    onehot_transf_fet = pd.get_dummies(df_copy[onehot_enc_fet])

    df_copy = pd.concat([df_copy.drop(onehot_enc_fet, axis=1), onehot_transf_fet], axis=1)

    return df_copy

def column_reorder(df):
    df_copy = df.copy()

    order = ['cat0', 'cat1', 'cat2', 'cont0', 'cont1', 'cont2', 'cont3',
       'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10',
       'cont11', 'cont12', 'cont13', 'cat3_A', 'cat3_B', 'cat3_C',
       'cat3_D', 'cat4_A', 'cat4_B', 'cat4_C', 'cat4_D', 'cat5_A', 'cat5_B',
       'cat5_C', 'cat5_D', 'cat6_A', 'cat6_B', 'cat6_C', 'cat6_D', 'cat6_E',
       'cat6_G', 'cat6_H', 'cat6_I', 'cat7_A', 'cat7_B', 'cat7_C', 'cat7_D',
       'cat7_E', 'cat7_F', 'cat7_G', 'cat7_I', 'cat8_A', 'cat8_B', 'cat8_C',
       'cat8_D', 'cat8_E', 'cat8_F', 'cat8_G', 'cat9_A', 'cat9_B', 'cat9_C',
       'cat9_D', 'cat9_E', 'cat9_F', 'cat9_G', 'cat9_H', 'cat9_I', 'cat9_J',
       'cat9_K', 'cat9_L', 'cat9_M', 'cat9_N', 'cat9_O']

    return df_copy[order]
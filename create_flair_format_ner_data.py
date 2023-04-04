import json

import pandas as pd

df = pd.read_csv('./train_finer.csv')
df_result = pd.DataFrame(columns=['text', 'annotation'])

with open('./labels.json', 'r') as JSON:
    label_dict = json.load(JSON)

xbrl_label_dict = {v: k for k, v in label_dict.items()}

f_train = open('./train.txt', 'w')
tokens_length = []
for i, row in df.iterrows():
    tokens = eval(row['tokens'])
    tags = eval(row['ner_tags'])

    #if len(tokens) <= 50:

    for token_index in range(len(tokens)):
        f_train.write(tokens[token_index] + ' ' + xbrl_label_dict[tags[token_index]])
        f_train.write('\n')
        f_train.write('\n')

# print(set(sorted(tokens_length)))

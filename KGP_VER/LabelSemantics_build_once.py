import json
import re
import torch
import os
import copy
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, BertModel, AlbertModel, AlbertTokenizer
import time, datetime
from sklearn.metrics import precision_score, classification_report, f1_score, recall_score
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import AlbertConfig, AlbertModel,AlbertForTokenClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

base = './data/finer/'
base_path = 'bert-base-uncased'
train_path = 'train.csv'
dev_path = 'dev.csv'
test_path = 'test.csv'
tag_file = 'labels.json'

import pandas as pd
import json

data_directory = "./data/finer/"
trainDf = pd.read_csv(data_directory + "train.csv")
testDf = pd.read_csv(data_directory + "test.csv")
devDf = pd.read_csv(data_directory + "dev.csv")
output_directory = "./data/finer_most_frequent_20/"

labels = json.load(open(data_directory + "labels.json"))
inv_labels = {v: k for k, v in labels.items()}
print(inv_labels)
labels_count = {}
for idx, row in trainDf.iterrows():
    for tag in eval(row['numerals-tags']):
        if tag not in labels_count:
            labels_count[tag] = 0
        labels_count[tag] += 1

sorted_label_count = list(dict(sorted(labels_count.items(), key=lambda item: item[1])).items())
print(len(sorted_label_count))
# print(sorted_label_count)

# x =1/0

few_labels = sorted_label_count[-5:]

print('few labels', few_labels)

label2id = {}
id2label = {}
labelIdx = 1
for idx, (label, _) in enumerate(few_labels):
    # print(label)
    # x =1/0
    label2id['B-' + label] = labelIdx
    id2label[labelIdx] = 'B-' + label
    labelIdx += 1
    # label2id['I-' + label] = labelIdx
    # id2label[labelIdx] = 'I-' + label
    # labelIdx += 1

label2id['O'] = 0
id2label[0] = 'O'

all_tags = []
include = []
for idx, row in trainDf.iterrows():
    ner_tags = eval(row['ner_tags'])
    modified_tags = []
    for tag in ner_tags:
        label = inv_labels[tag]
        if label not in label2id:
            modified_tags.append(0)
        else:
            modified_tags.append(label2id[label])
    all_tags.append(modified_tags)
    if max(modified_tags) > 0:
        val = 1
    else:
        val = 0
    include.append(val)

trainDf['all_tags'] = all_tags
trainDf['include'] = include

all_tags = []
include = []
for idx, row in devDf.iterrows():
    ner_tags = eval(row['ner_tags'])
    modified_tags = []
    for tag in ner_tags:
        label = inv_labels[tag]
        if label not in label2id:
            modified_tags.append(0)
        else:
            modified_tags.append(label2id[label])
    all_tags.append(modified_tags)
    if max(modified_tags) > 0:
        val = 1
    else:
        val = 0
    include.append(val)

devDf['all_tags'] = all_tags
devDf['include'] = include

all_tags = []
include = []
for idx, row in testDf.iterrows():
    ner_tags = eval(row['ner_tags'])
    modified_tags = []
    for tag in ner_tags:
        label = inv_labels[tag]
        if label not in label2id:
            modified_tags.append(0)
        else:
            modified_tags.append(label2id[label])
    all_tags.append(modified_tags)
    if max(modified_tags) > 0:
        val = 1
    else:
        val = 0
    include.append(val)

testDf['all_tags'] = all_tags
testDf['include'] = include

train = trainDf[trainDf['include'] == 1]
dev = devDf[devDf['include'] == 1]
test = testDf[testDf['include'] == 1]

print("Train size", len(train), "Dev size", len(dev), "Test size", len(test))


def load_data(base, file_path):
    df = pd.read_csv(os.path.join(base, file_path))
    df['tokens'] = df['tokens'].apply(lambda x: eval(x))
    df['ner_tags'] = df['ner_tags'].apply(lambda x: eval(x))
    return df['tokens'], df['ner_tags']


# Based on label list, create a list of labels with B, I tags and O tag.
def trans2id(base, label_path):
    with open(os.path.join(base, label_path), 'r') as fp:
        labels = json.load(fp)
    short_labels = set()
    short_labels.add("Other")
    for label in labels:
        short_labels.add(label[2:])
    tag2id = labels
    id2tag = {idx: tag for tag, idx in labels.items()}
    return tag2id, id2tag, short_labels


def gen_features(sample_tokens, sample_labels, tokenizer, tag2id, max_length):
    batch_token_ids, batch_token_type_ids, batch_attention_masks, batch_tags, batch_lengths = [], [], [], [], []
    for sample_idx in range(len(sample_tokens)):
        # if sample_idx in [8634, 21659, 44487, 54298]:
        # 	continue
        # sentence = " ".join(sample_tokens[sample_idx])
        # batch_lengths.append(len(sentence))
        sample_token_ids, sample_tags, sample_token_type_ids, sample_attention_masks = [], [], [], []
        for token_idx in range(len(sample_tokens.iloc[sample_idx])):
            token = sample_tokens.iloc[sample_idx][token_idx]
            tokenized_token_ids = tokenizer(token, add_special_tokens=False)
            token_ids = tokenized_token_ids.input_ids
            token_type_ids = tokenized_token_ids.token_type_ids
            attention_masks = tokenized_token_ids.attention_mask
            sample_token_ids.extend(token_ids)
            sample_token_type_ids.extend(token_type_ids)
            sample_attention_masks.extend(attention_masks)
            for token_id in token_ids:
                # print(sample_labels.iloc[sample_idx])
                # print(len(sample_labels.iloc[sample_idx]), len(sample_tokens.iloc[sample_idx]))
                # print(sample_idx, token_idx)
                sample_tags.append(sample_labels.iloc[sample_idx][token_idx])
        assert len(sample_tags) == len(sample_token_ids)
        CLS_ID = tokenizer.vocab['[CLS]']
        SEP_ID = tokenizer.vocab['[SEP]']
        PAD_ID = tokenizer.vocab['[PAD]']
        sample_token_ids = [CLS_ID] + sample_token_ids + [SEP_ID]
        sample_tags = [0] + sample_tags + [0]
        batch_token_ids.append(sample_token_ids)
        batch_token_type_ids.append(sample_token_type_ids)
        batch_attention_masks.append(sample_attention_masks)
        batch_tags.append(sample_tags)
    # Pad, truncate and verify
    # Returns an np.array object of shape ( len(batch_size) x max_length ) that contains padded/truncated gold labels
    batch_token_ids = pad_sequences(
        sequences=batch_token_ids,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    batch_token_ids[np.where(batch_token_ids[:, -1] != PAD_ID)[0], -1] = SEP_ID
    # Pad/Truncate the rest tags/labels
    batch_tags = pad_sequences(
        sequences=batch_tags,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    batch_tags[np.where(batch_tags[:, -1] != PAD_ID)[0], -1] = 0
    batch_token_type_ids = pad_sequences(
        sequences=batch_token_type_ids,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    batch_token_type_ids[np.where(batch_token_type_ids[:, -1] != PAD_ID)[0], -1] = 0
    batch_attention_masks = pad_sequences(
        sequences=batch_attention_masks,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    batch_attention_masks[np.where(batch_attention_masks[:, -1] != PAD_ID)[0], -1] = 0
    # if len(sample_tokens[sample_idx]) >= max_len - 2:
    # 	label = sample_labels[sample_idx][0:max_len - 2]
    # else:
    # 	label = sample_labels[sample_idx]
    # label = [tag2id['O']] + label + [tag2id['O']]
    # if len(label) < max_len:
    # 	label = label + [tag2id['O']] * (max_len - len(label))
    # assert len(label) == max_len
    # batch_tags.append(label)
    # inputs = tokenizer(sample_tokens[sample_idx], max_length=max_len, padding='max_length', return_tensors='pt')
    # input_id, token_type_id, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
    # try:
    # 	assert len(input_id) < max_len
    # except AssertionError:
    # 	print(len(input_id), len(sample_tokens[sample_idx]), sample_tokens[sample_idx])
    # batch_input_ids.append(input_id)
    # batch_token_type_ids.append(token_type_id)
    # batch_attention_masks.append(attention_mask)
    return batch_token_ids, batch_token_type_ids, batch_attention_masks, batch_tags


max_len = 512
bs = 32
# tokenizer = BertTokenizer.from_pretrained(base_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(base_path)

# Extra for 30 labels review
tag2id, id2tag = label2id, id2label

# tag2id, id2tag, short_labels = trans2id(base, tag_file)

# train_tokens,train_labels = load_data(base, train_path)
train_tokens, train_labels = train['tokens'].apply(lambda x: eval(x)), train['all_tags']
train_ids, train_token_type_ids, train_attention_masks, train_tags = gen_features(train_tokens, train_labels, tokenizer,
                                                                                  tag2id, max_len)

# dev_tokens,dev_labels = load_data(base,dev_path)
dev_tokens, dev_labels = dev['tokens'].apply(lambda x: eval(x)), dev['all_tags']
dev_ids, dev_token_type_ids, dev_attention_masks, dev_tags = gen_features(dev_tokens, dev_labels, tokenizer, tag2id,
                                                                          max_len)
print('data processing done')


def divideLabelIntoWords(label):
    return " ".join([s for s in re.split("([A-Z][^A-Z]*)", label)])


def build_label_representation(tag2id):
    # print("Before building label representations", torch.cuda.memory_summary())
    labels = []

    index_context = {
        "B": "begin word",
        "I": "middle word"
    }
    for k, v in tag2id.items():
        if k.split('-')[-1] != 'O':
            idx, label = k[:1], k[2:]
            # label = self.label_context[label]
            labels.append(index_context[idx] + " " + divideLabelIntoWords(label[8:]))
        else:
            labels.append("Other class")
    # print(labels)
    '''
    mutul(a,b) a和b维度是否一致的问题
    A.shape =（b,m,n)；B.shape = (b,n,k)
    torch.matmul(A,B) 结果shape为(b,m,k)
    '''
    tag_max_len = max([len(l) for l in labels])
    tag_embeddings = []
    encoder = BertModel.from_pretrained(base_path)
    # tokenizer = tokenizer.to('cuda')
    for idx, label in enumerate(labels):
        # print(idx, label)
        # print("While building label representations", idx, torch.cuda.memory_summary())
        input_ids = tokenizer.encode_plus(label, return_tensors='pt', padding='max_length', max_length=tag_max_len)
        outputs = encoder(input_ids=input_ids['input_ids'],
                          token_type_ids=input_ids['token_type_ids'],
                          attention_mask=input_ids['attention_mask'])

        # torch.cuda.empty_cache()
        pooler_output = outputs.pooler_output
        tag_embeddings.append(pooler_output)
        # print(pooler_output.shape)
        # print(len(tag_embeddings))

        # print(tag_embeddings)
    print(idx, label)
    label_embeddings = torch.stack(tag_embeddings, dim=0)
    label_embeddings = label_embeddings.squeeze(1)
    # print("After building label representations")
    return label_embeddings


label_representation = build_label_representation(tag2id)


class FewShot_NER(nn.Module):
    def __init__(self, base_model_path, tag2id, batch_size, tag_file):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_encoder = BertModel.from_pretrained(base_model_path)
        self.label_encoder = BertModel.from_pretrained(base_model_path)
        self.label_context = self.__read_file(tag_file)
        self.index_context = {
            "B": "begin word",
            "I": "middle word"
        }
        self.tokenizer = BertTokenizer.from_pretrained(base_path)
        # self.label_representation = self.build_label_representation(tag2id).to(self.device)
        self.batch_size = batch_size
        self.tag2id = tag2id

    def __read_file(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    def build_label_representation(self, tag2id):
        # print("Before building label representations", torch.cuda.memory_summary())
        labels = []
        for k, v in tag2id.items():
            if k.split('-')[-1] != 'O':
                idx, label = k[:1], k[2:]
                # label = self.label_context[label]
                labels.append(self.index_context[idx] + " " + divideLabelIntoWords(label[8:]))
            else:
                labels.append("Other class")
        # print(labels)
        '''
        mutul(a,b) a和b维度是否一致的问题
        A.shape =（b,m,n)；B.shape = (b,n,k)
        torch.matmul(A,B) 结果shape为(b,m,k)
        '''
        tag_max_len = max([len(l) for l in labels])
        tag_embeddings = []
        encoder = BertModel.from_pretrained(base_path)
        # tokenizer = tokenizer.to('cuda')
        for idx, label in enumerate(labels):
            # print(idx, label)
            # print("While building label representations", idx, torch.cuda.memory_summary())
            input_ids = tokenizer.encode_plus(label, return_tensors='pt', padding='max_length', max_length=tag_max_len)
            outputs = encoder(input_ids=input_ids['input_ids'],
                              token_type_ids=input_ids['token_type_ids'],
                              attention_mask=input_ids['attention_mask'])

            # torch.cuda.empty_cache()
            pooler_output = outputs.pooler_output
            tag_embeddings.append(pooler_output)
            # print(pooler_output.shape)
            # print(len(tag_embeddings))

            # print(tag_embeddings)
        print(idx, label)
        label_embeddings = torch.stack(tag_embeddings, dim=0)
        label_embeddings = label_embeddings.squeeze(1)
        # print("After building label representations")
        return label_embeddings

    def forward(self, inputs, flag=True):
        # print("Start of forward", torch.cuda.memory_summary())
        #if flag:
        #    label_representation = self.build_label_representation(self.tag2id)
        #    self.label_representation = label_representation.detach()
        #else:
        #label_representation = self.label_representation
        print("Label representation building complete")
        outputs = self.token_encoder(input_ids=inputs['input_ids'],
                                     token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask'])
        token_embeddings = outputs.last_hidden_state
        tag_lens, hidden_size = label_representation.shape
        current_batch_size = token_embeddings.shape[0]
        label_embedding = label_representation.expand(current_batch_size, tag_lens, hidden_size)
        label_embeddings = label_embedding.transpose(2, 1)
        matrix_embeddings = torch.matmul(token_embeddings.to(self.device), label_embeddings.to(self.device))
        softmax_embedding = nn.Softmax(dim=-1)(matrix_embeddings)
        label_indexs = torch.argmax(softmax_embedding, dim=-1)
        return matrix_embeddings, label_indexs


def trans2label(id2tag, data, lengths):
    new = []
    for i, line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]
        new.append(tmp)
    return new


def get_entities(tags):
    start, end = -1, -1
    prev = 'O'
    entities = []
    n = len(tags)
    tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
    for i, tag in enumerate(tags):
        if tag != 'O':
            if prev == 'O':
                start = i
                prev = tag
            elif tag == prev:
                end = i
                if i == n - 1:
                    entities.append((start, i))
            else:
                entities.append((start, i - 1))
                prev = tag
                start = i
                end = i
        else:
            if start >= 0 and end >= 0:
                entities.append((start, end))
                start = -1
                end = -1
                prev = 'O'
    return entities


def measure(preds, trues):
    flattened_seq_y_pred_str = []
    flattened_seq_y_true_str = []
    for pred_row, true_row in zip(preds, trues):
        for pred, true in zip(pred_row, true_row):
            if true != "O" or pred != "O":
                flattened_seq_y_pred_str.append(pred)
                flattened_seq_y_true_str.append(true)
    print(classification_report(flattened_seq_y_pred_str, flattened_seq_y_true_str))
    print("Micro: ",
          precision_recall_fscore_support(flattened_seq_y_pred_str, flattened_seq_y_true_str, average='micro'))
    print("Macro: ",
          precision_recall_fscore_support(flattened_seq_y_pred_str, flattened_seq_y_true_str, average='macro'))
    print("Weighted: ",
          precision_recall_fscore_support(flattened_seq_y_pred_str, flattened_seq_y_true_str, average='weighted'))
    return


for idx, train_tag in enumerate(train_tags):
    res = []
    for j in train_tag:
        if j == 'O':
            res.append(0)
        else:
            res.append(j)
    train_tags[idx] = train_tag

train_ids = torch.tensor([item for item in train_ids]).squeeze()
train_tags = torch.tensor(train_tags)
train_masks = torch.tensor([item for item in train_attention_masks]).squeeze()
train_token_type_ids = torch.tensor([item for item in train_token_type_ids]).squeeze()
# print(train_ids.shape,train_tags.shape,train_masks.shape,train_token_type_ids.shape)

dev_ids = torch.tensor([item for item in dev_ids]).squeeze()
dev_tags = torch.tensor(dev_tags)
dev_masks = torch.tensor([item for item in dev_attention_masks]).squeeze()
dev_token_type_ids = torch.tensor([item for item in dev_token_type_ids]).squeeze()

train_data = TensorDataset(train_ids, train_masks, train_token_type_ids, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(dev_ids, dev_masks, dev_token_type_ids, dev_tags)
valid_sampler = RandomSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

# print("Data built", torch.cuda.memory_summary())

fewshot = FewShot_NER(base_path, tag2id, bs, os.path.join(base, tag_file))

optimizer = torch.optim.Adam(fewshot.parameters(),
                             lr=5e-5  # default is 5e-5
                             # eps = 1e-8 # default is 1e-8
                             )

epochs = 1
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
fewshot.to(device)

max_grad_norm = 1.0
F1_score = 0

loss_function = CrossEntropyLoss()

# def define_loss_function(input,target):

# 	logsoftmax_func=nn.LogSoftmax(dim=1)
# 	logsoftmax_output=logsoftmax_func(input)

# 	nllloss_func=nn.NLLLoss()
# 	nlloss_output=nllloss_func(logsoftmax_output,target)
# 	return nlloss_output

tra_loss, steps = 0.0, 0

scaler = torch.cuda.amp.GradScaler()
for i in range(epochs):
    print('Epoch', i)
    # if i == 1:
    #	break
    fewshot.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        input_ids, masks, token_type_ids, labels = (i.to(device) for i in batch)

        # print(input_ids)
        matrix_embeddings, label_indexs = fewshot(
            {"input_ids": input_ids, "attention_mask": masks, "token_type_ids": token_type_ids})
        loss = loss_function(matrix_embeddings.view(-1, len(tag2id)),
                             torch.tensor(labels.type(torch.LongTensor)).cuda().view(-1))  # CrossEntropyLoss
        optimizer.zero_grad()
        loss.backward()
        tra_loss += loss
        steps += 1
        grad = torch.nn.utils.clip_grad_norm_(parameters=fewshot.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        if step % 30 == 0:
            print("epoch :{},step :{} ,Train loss: {}".format(i, step, tra_loss / steps))
    print("Training Loss of epoch {}:{}".format(i, tra_loss / steps))
    fewshot.eval()
    predictions, true_labels = [], []
    for step, batch in enumerate(tqdm(valid_dataloader)):
        input_ids, masks, token_type_ids, labels = (i.to(device) for i in batch)
        with torch.no_grad():
            matrix_embeddings, output_indexs = fewshot(
                {"input_ids": input_ids, "attention_mask": masks, "token_type_ids": token_type_ids}, flag=False)
        # scores = scores.detach().cpu().numpy()
        predictions.extend(output_indexs.detach().cpu().numpy().tolist())
        true_labels.extend(labels.to('cpu').numpy().tolist())
    #         lengths = lengths.detach().cpu().numpy().tolist()
    #     dev_lengths = dev_lengths.detach().cpu().numpy()
    # measure(predictions, true_labels)
# print('epoch {} : Acc : {},Recall : {},F1 :{}'.format(i,precision,recall,f1))
# if F1_score < f1:
#	F1_score = f1
#	torch.save(fewshot.state_dict(), 'save_models/model_{}_{}.pth'.format(i,F1_score))

# test_tokens,test_labels = load_data(base,test_path)
test_tokens, test_labels = test['tokens'].apply(lambda x: eval(x)), test['all_tags']
test_ids, test_token_type_ids, test_attention_masks, test_tags = gen_features(test_tokens, test_labels, tokenizer,
                                                                              tag2id, max_len)

test_ids = torch.tensor([item for item in test_ids]).squeeze()
test_tags = torch.tensor(test_tags)
test_masks = torch.tensor([item for item in test_attention_masks]).squeeze()
test_token_type_ids = torch.tensor([item for item in test_token_type_ids]).squeeze()

test_data = TensorDataset(test_ids, test_masks, test_token_type_ids, test_tags)
# test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=bs)

fewshot.eval()
test_pre, test_true = [], []
for idx, batch in enumerate(tqdm(test_dataloader)):
    # if idx == 1:
    #	break
    input_ids, masks, token_type_ids, labels = (i.to(device) for i in batch)

    with torch.no_grad():
        matrix_embeddings, output_indexs = fewshot(
            {"input_ids": input_ids, "attention_mask": masks, "token_type_ids": token_type_ids}, flag=False)

    test_pre.extend(output_indexs.detach().cpu().numpy().tolist())
    test_true.extend(labels.to('cpu').numpy().tolist())
print('test data result')
measure(test_pre, test_true)

import csv

with open("xbrl_desc_true.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test_true)
with open("xbrl_desc_pred.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test_pre)

# print('Test Acc : {},Recall : {},F1 :{}'.format(test_precision,test_recall,test_f1))

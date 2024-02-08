import json
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from tqdm import tqdm


### Read data
data, metrics, id_dict = [], [], {}
with open('user.json') as f:
    d = json.load(f)
    for i, dd in enumerate(d):
        data.append(dd['description'])
        metrics.append(list(dd['public_metrics'].values()))
        id_dict[dd['id']] = i
metrics = torch.tensor(StandardScaler().fit_transform(metrics))

### Create features
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')['input_ids']

config = BertConfig.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config).cuda()

batch_size = 512
total_size = int(len(data) / batch_size) + 1 if len(data) % batch_size > 0 else int(len(data) / batch_size)
total_feats = []
with torch.no_grad():
    for i in tqdm(range(total_size)):
        input_ids = data[i * batch_size:(i + 1) * batch_size].cuda()
        outputs = bert_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        out_embs = torch.mean(hidden_states[0], dim=1).cpu()
        total_feats.append(out_embs)

with open('X.pkl', 'wb') as handle:
    pickle.dump(torch.cat([metrics, torch.cat(total_feats, dim=0)], dim=-1), handle, protocol=pickle.HIGHEST_PROTOCOL)

### Create edges
df = pd.read_csv('edge.csv')
edge_index = []
for i, j in tqdm(df[df['relation'].isin(['followers', 'following'])][['source_id', 'target_id']].values):
    try:
        edge_index.append([id_dict[i], id_dict[j]])
    except:
        pass

with open('edge_index.pkl', 'wb') as handle:
    pickle.dump(torch.tensor(edge_index).T, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Create labels
df = pd.read_csv('label.csv')
label_dict = {}
for k, v in df[['id', 'label']].values:
    label_dict[k] = 0 if v == 'human' else 1

y = []
for k in id_dict.keys():
    y.append(label_dict[k])
y = np.array(y)

with open('y.pkl', 'wb') as handle:
    pickle.dump(torch.tensor(y), handle, protocol=pickle.HIGHEST_PROTOCOL)

### Create split
df = pd.read_csv('split.csv')
mask_dict = {}
mask_dict['train'] = np.array([id_dict[i] for i in df[df['split'] == 'train']['id'].values])
mask_dict['val'] = np.array([id_dict[i] for i in df[df['split'] == 'val']['id'].values])
mask_dict['test'] = np.array([id_dict[i] for i in df[df['split'] == 'test']['id'].values])

with open('split.pkl', 'wb') as handle:
    pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
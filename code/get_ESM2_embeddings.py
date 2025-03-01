import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


dataset_name_list = ['Dataset1', 'Dataset2', 'Dataset3']

pre_train = 'ESM2'

for dataset_name in dataset_name_list:
    phage_info_path = f'../data/{dataset_name}/rbp_embeddings_labels.csv'
    rbp_all_info = pd.read_csv(phage_info_path, low_memory=False)

    protein_sequences = rbp_all_info['Protein Sequence']

    model_directory_path = f'../pretrain_model/{pre_train}/'

    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    embedder = AutoModel.from_pretrained(model_directory_path).to(device)

    embeddings = []
    cnt = 1
    for protein in protein_sequences:
        print(f'The {cnt} protein sequence is being embedded, the length of which is: {len(protein)}')
        cnt += 1
        inputs = tokenizer(protein, return_tensors='pt').to(device)  # Tokenize the sequence
        with torch.no_grad():
            embedding = embedder(input_ids=inputs['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[1:-1]
            embedding = np.mean(embedding, axis=0)
            embeddings.append(embedding)

    embedding_df = pd.DataFrame(embeddings, columns=[f'{i+1}' for i in range(len(embedding))])

    rbp_labels = rbp_all_info['Host']
    rbp_embeddings_labels = pd.concat([embedding_df, rbp_labels], axis=1)

    label_encoder = LabelEncoder()
    rbp_embeddings_labels['Host'] = label_encoder.fit_transform(rbp_embeddings_labels['Host'])


    rbp_embeddings_labels_path = f'../data/{dataset_name}/rbp_embeddings_labels_{pre_train}.csv'
    rbp_embeddings_labels.to_csv(rbp_embeddings_labels_path, index=False)






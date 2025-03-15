import pandas as pd
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from sklearn.preprocessing import LabelEncoder

dataset_name_list = ['Dataset1', 'Dataset2', 'Dataset3']

pre_train = 'ProtT5'

for dataset_name in dataset_name_list:
    phage_info_path = f'../data/{dataset_name}/rbp_embeddings_labels.csv'
    rbp_all_info = pd.read_csv(phage_info_path, low_memory=False)

    protein_sequences = rbp_all_info['Protein Sequence']

    model_directory_path = f'../pretrain_model/{pre_train}/'
    embedder = ProtTransT5XLU50Embedder(device="cuda:0", model_directory=model_directory_path)

    cnt = 1
    embeddings = []
    for protein in protein_sequences:
        embedding = embedder.reduce_per_protein(embedder.embed(protein))
        print(f'The {cnt} protein sequence is being embedded, the length of which is: {len(protein)}')
        cnt += 1
        embeddings.append(embedding)

    embedding_df = pd.DataFrame(embeddings, columns=[f'{i + 1}' for i in range(len(embedding))])

    columns_host = ['Host Phylum', 'Host Class', 'Host Order', 'Host Family', 'Host Genus']
    if dataset_name == 'Dataset3':
        columns_host = ['Host Phylum', 'Host Class', 'Host Order', 'Host Family', 'Host Genus', 'Host Species']

    host_df = rbp_all_info[columns_host]
    rbp_embeddings_labels = pd.concat([host_df, embedding_df], axis=1)

    le = LabelEncoder()
    for column in columns_host:
        if column in rbp_embeddings_labels.columns:
            rbp_embeddings_labels[column] = le.fit_transform(rbp_embeddings_labels[column])

    rbp_embeddings_labels_path = f'../data/{dataset_name}/rbp_embeddings_labels_{pre_train}.csv'
    rbp_embeddings_labels.to_csv(rbp_embeddings_labels_path, index=False)



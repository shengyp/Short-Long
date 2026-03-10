import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


def read_embedding_data(file_path):
    print(f"正在加载数据: {file_path} ...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embeddings_raw = item['embeddings']

        if isinstance(embeddings_raw, np.ndarray):
            embeddings_tensor = torch.from_numpy(embeddings_raw).float()
        elif isinstance(embeddings_raw, list):
            if len(embeddings_raw) > 0 and isinstance(embeddings_raw[0], torch.Tensor):
                embeddings_tensor = torch.stack(embeddings_raw)
            else:
                embeddings_tensor = torch.tensor(np.array(embeddings_raw), dtype=torch.float)
        else:
            embeddings_tensor = torch.zeros(0, 768, dtype=torch.float)

        label = int(item['label'])
        return embeddings_tensor, label


def collate_fn_embeddings(batch):
    raw_embeddings, labels = zip(*batch)
    batch_size = len(raw_embeddings)

    # 健壮的长度获取
    post_counts = [len(e) for e in raw_embeddings]
    max_num_posts = max(post_counts) if post_counts else 0

    if batch_size > 0 and max_num_posts > 0:
        embedding_dim = raw_embeddings[0].shape[1]
    else:
        embedding_dim = 768

    padded_embeddings = torch.zeros(batch_size, max_num_posts, embedding_dim, dtype=torch.float)
    post_masks = torch.zeros(batch_size, max_num_posts, dtype=torch.float)

    for i, embs_tensor in enumerate(raw_embeddings):
        num_posts = len(embs_tensor)
        if num_posts > 0:
            padded_embeddings[i, :num_posts, :] = embs_tensor
            post_masks[i, :num_posts] = 1.0

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, labels, post_masks
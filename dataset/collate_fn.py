import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    emb1_list, emb2_list, label_list = zip(*batch)

    # remove extra batch dim [1, L, 320] → [L, 320]
    emb1_list = [e.squeeze(0) for e in emb1_list]
    emb2_list = [e.squeeze(0) for e in emb2_list]

    # pad each list separately
    emb1_padded = pad_sequence(emb1_list, batch_first=True)   # [B, Lmax1, 320]
    emb2_padded = pad_sequence(emb2_list, batch_first=True)   # [B, Lmax2, 320]

    labels = torch.tensor(label_list, dtype=torch.float)
    return emb1_padded, emb2_padded, labels

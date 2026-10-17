import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_token(token_list, keep_chars=""):
    pattern = rf"[^\w{re.escape(keep_chars)}]"
    normalized = []
    for token in token_list:
        if token.lower() == "<x>":
            normalized.append("<X>")
            continue
        token = re.sub(pattern, "", token)
        if not token:
            continue
        if "=" in token:
            parts = token.split("=")
            left = parts[0]
            left = re.sub(pattern, "", left)
            if left:
                normalized.append(left)
            normalized.append("<=>")
            right = parts[1] if len(parts) > 1 else ""
            right = re.sub(pattern, "", right)
            if right:
                if re.fullmatch(r"\d+", right):
                    normalized.append("<X>")
                else:
                    right = re.sub(r"\d+", "<X>", right)
                    normalized.append(right)
            continue

        if re.fullmatch(r"\d+", token):
            normalized.append("<X>")
        elif re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", token):
            normalized.append("<X>")
        else:
            token = re.sub(r"\d+", "<X>", token)
            normalized.append(token)
    return normalized


def find_sorted_similar(edge_embeddings, target_index=0):
    target = edge_embeddings[target_index]
    cos = F.cosine_similarity(target.unsqueeze(0), edge_embeddings)
    values, indices = torch.sort(cos, descending=True)
    return [
        (idx, score)
        for idx, score in zip(indices.tolist(), values.tolist())
        if idx != target_index
    ]


def nonvariable_overlap(left, right):
    length = max(len(left), len(right))
    matches = sum(a == b and "<x>" not in a.lower() for a, b in zip(left, right))
    return matches / length if length else 0.0


def get_topk_similar(
    result_,
    log_list,
    k=3,
    s_t=0.8,
    target_index=0,
    min_overlap=0.2,
    structure_logs=None,
):
    if structure_logs is None:
        structure_logs = log_list
    examples = []
    if k <= 0:
        return examples
    for idx, sim in result_:
        log = log_list[idx]
        if idx == target_index:
            continue
        overlap = nonvariable_overlap(structure_logs[target_index], structure_logs[idx])
        if overlap < min_overlap or overlap == 0:
            continue
        if log not in examples and sim >= s_t:
            print(log, sim)
            examples.append(log)
        if len(examples) == k:
            break
    return examples


class MaskHyperGraphConv(nn.Module):

    def __init__(self, edge_lists, emb_dim=64, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.dim = emb_dim
        vocab = set()
        for e in edge_lists:
            vocab.update(e)
        self.vocab2id = {t: i for i, t in enumerate(vocab)}
        self.num_nodes = len(vocab)
        self.node_embedding = nn.Embedding(self.num_nodes, emb_dim)
        self.node_to_edge = nn.Linear(emb_dim, emb_dim)
        self.edge_to_node = nn.Linear(emb_dim, emb_dim)
        self.edge_project = nn.Linear(emb_dim, emb_dim)
        self.mask_ids = set()
        for token, idx in self.vocab2id.items():
            if self.is_unlearn_token(token):
                self.mask_ids.add(idx)
        self.register_buffer("H", self.build_incidence(edge_lists))

    def is_unlearn_token(self, t):
        if t.lower() == "<x>":
            return True
        core = re.sub(r"<x>", "", t.lower())
        if re.fullmatch(r"[^\w]+", core):
            return True
        return False

    def build_incidence(self, edge_lists):
        H = torch.zeros(self.num_nodes, len(edge_lists))
        for eid, edge in enumerate(edge_lists):
            for token in set(edge):
                nid = self.vocab2id[token]
                H[nid, eid] = 1
        return H

    def mask_nodes(self, X):
        mask = torch.ones(X.size(0), device=X.device)
        for idx in self.mask_ids:
            mask[idx] = 0
        return X * mask.unsqueeze(-1)

    def forward(self):
        device = self.H.device
        H = self.H
        node_ids = torch.arange(self.num_nodes, device=device)
        X = self.node_embedding(node_ids)
        X = self.mask_nodes(X)
        edge_degree = H.sum(dim=0)
        edge_degree[edge_degree == 0] = 1
        E = torch.matmul(H.T, X)
        E = E / edge_degree.unsqueeze(-1)
        E = torch.relu(self.node_to_edge(E))
        node_degree = H.sum(dim=1)
        node_degree[node_degree == 0] = 1
        X_new = torch.matmul(H, E)
        X_new = X_new / node_degree.unsqueeze(-1)
        X_new = torch.relu(self.edge_to_node(X_new))
        X_new = self.mask_nodes(X_new)
        E_final = torch.matmul(H.T, X_new)
        E_final = E_final / edge_degree.unsqueeze(-1)
        E_final = torch.relu(self.edge_project(E_final))
        return E_final


def build_positive_pairs(sub_structure_id):
    n = len(sub_structure_id)
    positive = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # target hyperedge
            if i == 0:
                positive[i].add(j)
            elif (
                sub_structure_id[i] != -1 and sub_structure_id[i] == sub_structure_id[j]
            ):
                positive[i].add(j)
    return positive


def contrastive_loss(embeddings, batch_ids, positive_pairs, temperature=0.1):
    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    losses = []
    for i, global_i in enumerate(batch_ids):
        positives = [j for j in batch_ids if j in positive_pairs[global_i]]
        if len(positives) == 0:
            continue
        mask = [j != global_i for j in batch_ids]
        denominator = torch.exp(sim[i][mask] / temperature).sum()
        for p in positives:
            p_index = batch_ids.index(p)
            numerator = torch.exp(sim[i][p_index] / temperature)
            losses.append(-torch.log(numerator / denominator))

    if len(losses) == 0:
        return embeddings.sum() * 0
    return torch.mean(torch.stack(losses))


def train_MAHGCN(
    model, edge_lists, sub_structure_id, epochs=10, batch_size=64, lr=0.01
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positive_pairs = build_positive_pairs(sub_structure_id)
    ids = list(range(len(edge_lists)))
    for epoch in range(epochs):
        total_loss = 0
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            optimizer.zero_grad()
            embeddings = model()
            batch_embeddings = embeddings[batch_ids]
            loss = contrastive_loss(batch_embeddings, batch_ids, positive_pairs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print("epoch:",epoch,"loss:",total_loss)
    return model


def get_example(
    log_list, dataset, sub_structure_id, log_index=0, min_overlap=0.5, s_t=0.8
):
    keep_chars_dict = {
        "Apache": ".:_/",
        "BGL": ".:_/",
        "Hadoop": ".:_/|",
        "HDFS": ".:_/",
        "HealthApp": ".:_/|",
        "HPC": ".:_/",
        "Linux": ".:_/",
        "Mac": ".:_/",
        "OpenSSH": ".:_/",
        "OpenStack": ".:_/",
        "Proxifier": ".:_/",
        "Spark": ".:_/",
        "Thunderbird": ".:_/",
        "Zookeeper": ".:_/",
    }
    sub_logs = [normalize_token(log, keep_chars_dict[dataset]) for log in log_list]
    model = MaskHyperGraphConv(sub_logs, emb_dim=64, num_layers=1)
    model = train_MAHGCN(
        model, sub_logs, sub_structure_id, epochs=3, batch_size=128, lr=0.1
    )
    with torch.no_grad():
        edge_embeddings = model()
    result = find_sorted_similar(edge_embeddings, log_index)
    print(log_list[log_index])
    print("\nsimilar logs:")
    examples = get_topk_similar(
        result,
        log_list,
        k=3,
        s_t=s_t,
        target_index=log_index,
        min_overlap=min_overlap,
        structure_logs=log_list,
    )
    return examples

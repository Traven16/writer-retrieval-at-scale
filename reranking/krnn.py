import torch


def krnn_reranking(X, k, gamma=1.0, r=True):
    X = torch.nn.functional.normalize(X.float(), dim=1)
    S = torch.mm(X, X.t())
    _, initial_rank = S.topk(k=S.shape[0], dim=-1, largest=True, sorted=True)
    kNN = initial_rank[:, 1 : k + 1]

    reranked = torch.zeros_like(X)
    step = max(0.0, min(1.0, float(gamma)))
    for i in range(X.shape[0]):
        feat = X[i]
        nn = kNN[i]
        if r:
            neighbors = [X[j] for j in nn if i in kNN[j]]
        else:
            neighbors = [X[j] for j in nn]

        if not neighbors:
            reranked[i] = feat
            continue

        neighborhood = torch.stack(neighbors, dim=0).mean(dim=0)
        # Move only part of the way toward the local neighborhood so repeated
        # iterations continue to refine the embedding instead of collapsing in one step.
        reranked[i] = feat + step * (neighborhood - feat)

    reranked = torch.nn.functional.normalize(reranked, dim=1)
    cosine_distance = 1 - torch.mm(reranked, reranked.t())
    return reranked.cpu(), cosine_distance.cpu()

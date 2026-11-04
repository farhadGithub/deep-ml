import numpy as np

def contrastive_loss(E1, E2, same, m):
    # mean pairwise contrastive loss, eq. 16.10
    E1 = np.array(E1, dtype=float)
    E2 = np.array(E2, dtype=float)
    same = np.array(same, dtype=bool)
    diff = np.sqrt(np.sum((E1-E2)**2, axis=1))
    diff[~same] = np.maximum(m-diff[~same],0)
    ans = np.mean(diff**2)
    return ans

def mine_triplets(E, labels, m, mode):
    # list of (anchor, positive, negative) index tuples
    E = np.array(E, dtype=float)
    labels = np.array(labels, dtype=int)
    n = E.shape[0]
    diff = np.sqrt(np.sum((E[:,None,:]-E[None,:,:])**2, axis=2))
    triplets = []
    for i in range(n):
        pos = np.where(labels==labels[i])[0]
        neg = np.where(labels!=labels[i])[0]
        if len(pos)<=1 or len(neg)==0:
            continue
        pos = pos[pos!=i]
        hardest_pos = pos[np.argmin(diff[i,pos])]
        if mode=='hard':
            for n in neg:
                if diff[i,n]<diff[i,hardest_pos]:
                    triplets.append((i, hardest_pos, n))
        elif mode=='semi-hard':
            for n in neg:
                if diff[i,n]>diff[i,hardest_pos] and diff[i,n]<diff[i,hardest_pos]+m:
                    triplets.append((i, hardest_pos, n))
        elif mode=='all':
            for n in neg:
                triplets.append((i, hardest_pos, n))
    return triplets

def batch_triplet_loss(E, triplets, m):
    # mean hinge triplet loss, eq. 16.11
    E = np.array(E, dtype=float)
    triplets = np.array(triplets, dtype=int)
    if len(triplets)==0:    
        return 0.0
    a = E[triplets[:,0],:]
    p = E[triplets[:,1],:]
    n = E[triplets[:,2],:]
    d_ap = np.sqrt(np.sum((a-p)**2, axis=1))
    d_an = np.sqrt(np.sum((a-n)**2, axis=1))
    loss = np.maximum(d_ap**2-d_an**2+m, 0)
    return float(np.mean(loss))
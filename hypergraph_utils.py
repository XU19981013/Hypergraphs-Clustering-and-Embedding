
import numpy as np


def topvec(matrix,k):  #输出前k小的特征值对应的特征向量
    e_vals, e_vecs = np.linalg.eig(matrix)     # 拉普拉斯矩阵热政治分解
    sorted_indices=np.argsort(e_vals)
    print("sorted:",sorted_indices)
    print("sorted:",sorted_indices[:k])
    return e_vals[sorted_indices[:k]],e_vecs[:,sorted_indices[1:k]]

def Eu_dis(x):
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):#连接多模态功能，返回融合后的特征矩阵

    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):

    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):

    H = np.array(H)
    print("HHH",H)
    n_edge = H.shape[1]
    # the weight of the hyperedge

    w_S = np.sum(H, axis=0)
    W = w_S
    print("Wshape",W.shape)

    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):#

    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    print("H",H.shape)
    print("H",n_obj,n_edge)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    print(K_neigs)
    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)  #多个H合并操作
        else:
            H.append(H_tmp)
    return H

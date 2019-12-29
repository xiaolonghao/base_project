"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid) 
reid/evaluation_metrics/ranking.py. Modifications: 
1) Only accepts numpy data input, no torch is involved.
1) Here results of each query can be returned.
2) In the single-gallery-shot evaluation case, the time of repeats is changed 
   from 10 to 100.
"""
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat,query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, topk=100,
    separate_camera_set=False, single_gallery_shot=False, first_match_break=False,average=True):

    '''
    Args:
        distmat: numpy array with shape [num_query, num_gallery], the  pairwise distance between query and gallery samples
        query_ids: numpy array with shape [num_query]
        gallery_ids: numpy array with shape [num_gallery]
        query_cams: numpy array with shape [num_query]
        gallery_cams: numpy array with shape [num_gallery]
        average: whether to average the results across queries
    Returns:
        If `average` is `False`:
            ret: numpy array with shape [num_query, topk]
            is_valid_query: numpy array with shape [num_query], containing 0's and 
            1's, whether each query is valid or not
        If `average` is `True`:
            numpy array with shape [topk]
    '''
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros([m, topk])
    is_valid_query = np.zeros(m)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / num_valid_queries
    return ret, is_valid_query

def mean_ap( distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, average=True):
    """
    Args:
        distmat: numpy array with shape [num_query, num_gallery], the  pairwise distance between query and gallery samples
        query_ids: numpy array with shape [num_query]
        gallery_ids: numpy array with shape [num_gallery]
        query_cams: numpy array with shape [num_query]
        gallery_cams: numpy array with shape [num_gallery]
        average: whether to average the results across queries
    Returns:
        If `average` is `False`:
            ret: numpy array with shape [num_query]
        is_valid_query: numpy array with shape [num_query], containing 0's and 1's, whether each query is valid or not
        If `average` is `True`:
            a scalar
    """

    # -------------------------------------------------------------------------
    # The behavior of method `sklearn.average_precision` has changed since version
    # 0.19.
    # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
    # (https://github.com/zhunzhong07/person-re-ranking/
    # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
    # (http://www.liangzheng.org/Project/project_reid.html).
    # My current awkward solution is sticking to this older version.
    import sklearn
    cur_version = sklearn.__version__
    required_version = '0.18.1'
    if cur_version != required_version:
        print('User Warning: Version {} is required for package scikit-learn, '
              'your current version is {}. '
              'As a result, the mAP score may not be totally correct. '
              'You can try `pip uninstall scikit-learn` '
              'and then `pip install scikit-learn=={}`'.format( required_version, cur_version, required_version))

    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): 
            continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    """
    CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
    url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
    Matlab version: https://github.com/zhunzhong07/person-re-ranking

    q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
    q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
    g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]

    k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)

    Returns:
      final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
    """
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

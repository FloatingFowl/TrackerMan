import numpy as np


def sub(s, I):
    if len(I) > 0:
        n = [i for i in s]
        for i in range(len(n)):
            f = n[i]
            s[f] = s[f][I]
    return s


def tracking_dp(dres, c_en, c_ex, c_ij, betta,
                thr_cost, max_it, nms_in_loop):
    '''
    Non max suppression is not implemented
    '''
    if max_it is None:
        max_it = 10e5;
    if thr_cost is None:
        thr_cost = 0;

    thr_nms = 0.5
    dnum = len(dres['x'])
    dres['c'] = betta - dres['r']

    min_c = -np.inf
    min_cs = np.zeros(int(10e5))
    inds = np.zeros(int(10e5))
    it = 0  # zero in matlab
    k = 0  # zero in matlab
    inds_all = np.zeros(int(10e5))
    id_s = np.zeros(int(10e5))
    redo_nodes = list(range(dnum))

    dres['dp_c'] = np.array([0. for i in redo_nodes])
    dres['dp_link'] = np.array([0. for i in redo_nodes])
    dres['orig'] = np.array([0. for i in redo_nodes])

    while min_c < thr_cost and it < max_it:
        it = it + 1

        dres['dp_c'][redo_nodes] = dres['c'][redo_nodes] + c_en
        dres['dp_link'][redo_nodes] = -1
        dres['orig'][redo_nodes] = redo_nodes

        # print(dres['dp_link'][0:10])
        # print(dres['orig'][0:10])

        for ii in range(len(redo_nodes)):
            i = redo_nodes[ii]
            f2 = dres['pr'][i]
            if len(f2) == 0:
                continue

            costs = c_ij + dres['c'][i] + dres['dp_c'][f2]
            min_cost, j = np.min(costs), np.argmin(costs)
            min_link = f2[j]

            if dres['dp_c'][i] > min_cost:
                dres['dp_c'][i] = min_cost
                dres['dp_link'][i] = min_link
                dres['orig'][i] = dres['orig'][min_link]

        min_c = np.min(dres['dp_c'] + c_ex)
        ind = np.argmin(dres['dp_c'] + c_ex)
        inds = np.zeros(dnum).astype('int32')

        k1 = -1  # zero in matlab
        while not ind == -1:
            k1 = k1 + 1
            inds[k1] = ind
            ind = dres['dp_link'][int(ind)]

        inds = inds[:k1 + 1]
        inds_all[k:k + len(inds)] = inds
        id_s[k:k + len(inds)] = it
        k = k + len(inds)

        # if nms_in_loop:
        #    supp_inds = nms_aggressive(dres, inds, thr_nms)
        #    origs = np.unique(dres['orig'][supp_inds])
        #    redo_nodes = np.where(dres['orig'] == origs)
        # else:
        supp_inds = inds
        try:
            origs = inds[-1]
        except:
            origs = 0
        redo_nodes = np.where(dres['orig'] == origs)

        redo_nodes = np.setdiff1d(redo_nodes, supp_inds)
        dres['dp_c'][supp_inds] = np.inf
        dres['c'][supp_inds] = np.inf

        min_cs[it] = min_c

    inds_all = inds_all[:k]
    id_s = id_s[:k]

    # line 83
    res = sub(dres, inds_all.astype('int32'))

    return res, min_cs

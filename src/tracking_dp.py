import numpy as np
import logging

def sub(s, I):
    if len(I) > 0:
        n = [i for i in s]
        for i in range(len(n)):
            f = n[i]
            s[f] = s[f][I]
    return s


def tracking_dp(ndct, c_en, c_ex, c_ij, betta,
                threshold_cost, max_iterations, nms_in_loop):
    '''
    Non max suppression is not implemented
    '''

    ndum = len(ndct['x'])
    redo_nodes = list(range(ndum))
    ndct['dp_c'] = np.array([0. for i in redo_nodes])
    ndct['dp_link'] = np.array([0. for i in redo_nodes])
    ndct['orig'] = np.array([0. for i in redo_nodes])


    if max_iterations is None:
        max_iterations = 10e5;
    if threshold_cost is None:
        threshold_cost = 0;

    # thr_nms = 0.5
    ndct['c'] = betta - ndct['r']

    min_c = -np.inf
    min_cs = np.zeros(int(10e5))

    iteration = 0
    k = 0
    inds_all = np.zeros(int(10e5))
    id_s = np.zeros(int(10e5))

    while min_c < threshold_cost and iteration < max_iterations:
        iteration = iteration + 1

        ndct['dp_c'][redo_nodes] = ndct['c'][redo_nodes] + c_en
        ndct['dp_link'][redo_nodes] = -1
        ndct['orig'][redo_nodes] = redo_nodes

        for ii in range(len(redo_nodes)):
            i = redo_nodes[ii]
            f2 = ndct['pr'][i]
            if len(f2) == 0:
                continue

            costs = c_ij + ndct['c'][i] + ndct['dp_c'][f2]
            min_cost, j = np.min(costs), np.argmin(costs)
            min_link = f2[j]

            if ndct['dp_c'][i] > min_cost:
                ndct['dp_c'][i] = min_cost
                ndct['dp_link'][i] = min_link
                ndct['orig'][i] = ndct['orig'][min_link]

        min_c = np.min(ndct['dp_c'] + c_ex)
        ind = np.argmin(ndct['dp_c'] + c_ex)
        inds = np.zeros(ndum).astype('int32')

        k1 = -1
        while not ind == -1:
            k1 = k1 + 1
            inds[k1] = ind
            ind = ndct['dp_link'][int(ind)]

        inds = inds[:k1 + 1]
        inds_all[k:k + len(inds)] = inds
        id_s[k:k + len(inds)] = iteration
        k = k + len(inds)

        # TODO: Should debug this part
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
        redo_nodes = np.where(ndct['orig'] == origs)

        redo_nodes = np.setdiff1d(redo_nodes, supp_inds)
        ndct['dp_c'][supp_inds] = np.inf
        ndct['c'][supp_inds] = np.inf

        min_cs[iteration] = min_c

    inds_all = inds_all[:k]
    id_s = id_s[:k]

    res = sub(ndct, inds_all.astype('int32'))

    return res, min_cs

if __name__ == "__main__":
    import cv2, numpy as np, pickle
    import os, shutil
    from importlib import reload
    import grapher

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info("Loading pedestrian detections cache")

    with open('pedestrian_detections.pkl', 'rb') as f:
        dct = pickle.load(f)

    reload(grapher)
    ndct = grapher.graphMaker(dct);

    logging.info("Tracking objects in the video")
    res, min_cs = tracking_dp(ndct, 10., 10., 0, 0.2, 18, np.inf, False)
    logging.info("Tracking complete")

    def drawRect(img, x, y, w, h):
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img


    logging.info("Drawing bounding boxes on images")
    shutil.copytree('seq03-img-left/', 'results/')
    images_folder = 'results/'
    images = sorted(os.listdir(images_folder))

    images = [cv2.imread(os.path.join(images_folder, image)) for image in sorted(images)]

    for i in range(len(res['x'])):
        frame = res['fr'][i] - 1
        images[frame] = drawRect(images[frame], res['x'][i], res['y'][i], res['w'][i], res['h'][i])

    height, width, channels = images[0].shape

    logging.info("Writing results to tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter('tracking.mp4', fourcc, 20.0, (width, height))

    for image in images:
        out.write(image)

    out.release()

    logging.info("Deleting temporary folders")
    shutil.rmtree('results/')







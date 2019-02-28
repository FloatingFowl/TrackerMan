'''
Input: Dictionary containing values as [x, y, w, h, r=score, fr=frame number] format
Return: Same dictionary with adjacency lists for each node for incoming 
        edges' corresponding vertex
'''

import numpy as np

def graphMaker(detections):

    adjList = [[] for _ in range(np.sum(detections['fr']==1))]

    # bbox overlap, bbox size
    threshold = {
                'overlap'   : 0.5, 
                'size'      : 0.8
                } 
    
    for f in range(2, detections['fr'].max() + 1):
        
        # Possible TODO: add K for number of frames to connect across
        prev = np.where(detections['fr'] == f-1)[0]
        cur  = np.where(detections['fr'] == f)[0]

        for i in cur:

            # Possible TODO: When the number of correspondences is zero
            overlap_scores = computeOverlap(i, prev, detections)
            prev_ov = prev[np.where(overlap_scores > threshold['overlap'])[0]]
            size_scores = computeSizeChange(i, prev_ov, detections)
            prev_sc = prev_ov[np.where(size_scores > threshold['size'])[0]]
            adjList.append(prev_sc)

    detections['pr'] = np.array(adjList)
    return detections

def computeOverlap(i, arr, detections):

    overlap = np.zeros(arr.shape[0], dtype=float)

    # get corners and area for each point

    l1, r1 = detections['x'][i], detections['x'][i] + detections['w'][i] - 1
    l2, r2 = detections['x'][arr], detections['x'][arr] + detections['w'][arr] - 1
    b1, t1 = detections['y'][i], detections['y'][i] + detections['h'][i] - 1
    b2, t2 = detections['y'][arr], detections['y'][arr] + detections['h'][arr] - 1
    a1, a2 = detections['w'][i] * detections['h'][i], detections['w'][arr] * detections['h'][arr]

    # intersection 

    ml, mr = np.maximum(l1,l2), np.minimum(r1,r2)
    mb, mt = np.maximum(b1,b2), np.minimum(t1,t2)
    nw, nh = mr - ml + 1, mt - mb + 1
    
    inds = np.where((nw > 0) & (nh > 0))[0]
    intersection = nw[inds] * nh[inds]

    # intersection over union

    union = a1 + a2[inds] - intersection
    overlap[inds] = intersection / union
    return overlap

def computeSizeChange(i, arr, detections):

    h_difference = detections['h'][i] / detections['h'][arr]
    size_change = np.minimum(h_difference, np.reciprocal(h_difference))
    return size_change

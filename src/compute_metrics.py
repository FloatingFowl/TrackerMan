import numpy as np
import cv2, numpy as np, os, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grapher
import random
import functools
import make_detections
import os
import motmetrics as mm


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def box(x,y,w,h):
	return [x,y,x+w,y+h]

def match(gtfile,out,detections):
	'''Computes metrics'''
	acc = mm.MOTAccumulator(auto_id=True)
	gt=np.loadtxt(gtfile,delimiter=',')
	fin_list=[]
	for i in range(len(out)):
		for j in out[i]:
			fin_list.append([j,i])
	fin_list.sort()
	maxfr=max(gt[:,0])
	detptr=0
	gtptr=0
	print(maxfr)
	for i in range(1,int(maxfr)):
		#print(i)
		objs=[]
		inds=[]
		gtobjs=[]
		gtinds=[]
		while gtptr<len(gt):
			if gt[gtptr][0]>i:
				break
			if gt[gtptr][-4]>0:
				gtobjs.append(str(gt[gtptr][1]))
				gtinds.append(gtptr)
			gtptr+=1

		while detptr<len(detections['fr']) and detptr<len(fin_list):
			if detections['fr'][fin_list[detptr][0]]>i:
				break
			objs.append(fin_list[detptr][1])
			inds.append(fin_list[detptr][0])
			detptr+=1

		distmatrix=[[0 for i in range(len(objs))] for j in range(len(gtobjs))]
		for i in range(len(gtinds)):

			b1=box(gt[gtinds[i]][2],gt[gtinds[i]][3],gt[gtinds[i]][4],gt[gtinds[i]][5])

			for j in range(len(objs)):

				b2=box(detections['x'][inds[j]],detections['y'][inds[j]],detections['w'][inds[j]],detections['h'][inds[j]])
				iou=bb_intersection_over_union(b1,b2)
				if iou<0.5:
					distmatrix[i][j]=np.nan
				else:
					distmatrix[i][j]=1-iou
		acc.update(gtobjs,objs,distmatrix)
	mh = mm.metrics.create()
	summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
	#return summary[1],summary[2]
	print(summary)




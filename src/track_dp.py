import numpy as np
import cv2, numpy as np, os, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grapher
import random
import functools
import make_detections
import os

         
def colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        
    clrs= [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    random.shuffle(clrs)
    return clrs 

def dp_1(detections,c_in,c_ex,c_ij,beta,thr_cost,max_it):
    '''Takes detections in and returns the nodes which form the 1st track'''

    c_arr=[]
    for i in range(len(detections['r'])):
        c_arr.append(beta-detections['r'][i])
    detections['c']=c_arr

    d_num=len(detections['x'])

    print("number of detections",d_num)

    detections['dp_c']=[]
    detections['dp_link']=[]
    detections['orig']=[]

    min_c = 10000000000000000000000000000000000000000000
    it=0
    k=0
    inds_all=[0 for i in range(100000)]
    id_s =[0 for i in range(100000)]
    redo_nodes=[i for i in range(d_num)]

    for i in range(len(redo_nodes)):
        detections['r'][i]*=-1
        detections['dp_c'].append(detections['r'][i]+c_in)
        #print(detections['r'][i])
        detections['dp_link'].append(-1)
        
    print("base cases done")
    tot_min=10000000000000000
    tot_amin=-1
    print("DP starting")
    for i in range(len(redo_nodes)):
        
        mi,amin=c_in,-1
        for node in detections['pr'][i]:
            if detections['dp_c'][node]+detections['r'][i]<mi:
                mi=detections['dp_c'][node]+detections['r'][i]
                amin=node

        detections['dp_c'][i]=mi
        detections['dp_link'][i]=amin
        if mi < tot_min:
            tot_min=mi
            tot_amin=i


    final=[]
    cur_ind=tot_amin
    print("Getting path")
    while cur_ind!=-1:
        final.append(cur_ind)
        cur_ind=detections['dp_link'][cur_ind]
    return final


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

def dp_many(detections,c_in,c_ex,c_ij,beta,thr_cost,max_it):
    c_arr=[]
    for i in range(len(detections['r'])):
        c_arr.append(beta-detections['r'][i])
    detections['c']=c_arr
    paths=[]
    d_num=len(detections['x'])

    print("number of detections",d_num)

    detections['dp_c']=[]
    detections['dp_link']=[]
    detections['orig']=[]
    detections['free']=[]

    min_c = 10000000000000000000000000000000000000000000
    it=0
    k=0
    inds_all=[0 for i in range(100000)]
    id_s =[0 for i in range(100000)]
    redo_nodes=[i for i in range(d_num)]

    for i in range(len(redo_nodes)):
        detections['r'][i]*=-1
        detections['dp_c'].append(detections['r'][i]+c_in)
        #print(detections['r'][i])
        detections['dp_link'].append(-1)
        detections['free'].append(-1)
    print("base cases done")
    while 1:
        print("iteration",k)
        for i in range(len(redo_nodes)):
            detections['dp_c'][i]=detections['r'][i]+c_in
            detections['dp_link'][i]=-1

        k+=1
        tot_min=10000000000000000
        tot_amin=-1
        print("DP starting")
        for i in range(len(redo_nodes)):
            
            mi,amin=c_in,-1
            if detections['free'][i]>0:
                detections['dp_c'][i]=10000000000000000000000
            for node in detections['pr'][i]:
                if detections['dp_c'][node]+detections['r'][i]<mi and detections['free'][node]==-1:
                    mi=detections['dp_c'][node]+detections['r'][i]
                    amin=node

            detections['dp_c'][i]=mi
            detections['dp_link'][i]=amin
            if mi < tot_min and detections['free'][i]<0:
                tot_min=mi
                tot_amin=i
        print("cost is",tot_min)
        if tot_min>thr_cost:
            break
        final=[]
        cur_ind=tot_amin
        if detections['free'][cur_ind]>0:
            break
        print("Getting path")
        while cur_ind!=-1:
            final.append(cur_ind)
            detections['free'][cur_ind]=1
            
            #paths.append(final)
            x1,y1=detections['x'][cur_ind],detections['y'][cur_ind]
            x2,y2=x1+detections['w'][cur_ind],y1+detections['h'][cur_ind]
            boxa=[x1,y1,x2,y2]
            for i in range(cur_ind+1,len(detections['x'])):
                if detections['fr'][i]>detections['fr'][cur_ind]:
                    break
                x1,y1=detections['x'][i],detections['y'][i]
                x2,y2=x1+detections['w'][i],y1+detections['h'][i]
                boxb=[x1,y1,x2,y2]
                if bb_intersection_over_union(boxa,boxb)>0 and abs(detections['r'][i])<abs(detections['r'][cur_ind]):
                    detections['free'][i]=1
                    #print('done')

            for i in range(cur_ind-1,-1,-1):
                if detections['fr'][i]<detections['fr'][cur_ind]:
                    break

                x1,y1=detections['x'][i],detections['y'][i]
                x2,y2=x1+detections['w'][i],y1+detections['h'][i]
                boxb=[x1,y1,x2,y2]
                if bb_intersection_over_union(boxa,boxb)>0 and abs(detections['r'][i])<abs(detections['r'][cur_ind]):
                    detections['free'][i]=1
                    #print('done')
            cur_ind=detections['dp_link'][cur_ind]
        paths.append(final)
    return paths



def convert(i):
    ti=str(i)
    no=6-len(ti)
    return no*"0" + ti

def drawRect(img,x,y,w,h,color=(255,0,0)):
    x1,y1=int(x),int(y)
    x2,y2=int(x+w),int(y+h)
    cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
    return img

def customsort(a,b):
    return a[0]<b[0]


if __name__=="__main__":
    

    base='../2DMOT2015/train/PETS09-S2L1/'
    seqname=base.split('/')[-2]
    f=base+'det/det.txt'
    dct = make_detections.make_detections(f)
    ndct = grapher.graphMaker(dct)
    
    detections=ndct
    print('length is',len(detections['x']))
    
    out=dp_many(ndct,10,10,0,0.2,0,10000000)
    out=np.array(out)
    print('totals is',out.shape)
    fin_list=[]
    for i in range(len(out)):
        for j in out[i]:
            fin_list.append([j,i])
    no_colors=len(out)+1
    colors = [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(no_colors)]
    print("total is",len(fin_list))
    #fin_list=list(set(fin_list))
    fin_list.sort()
    print('finally it is',len(fin_list))

    cur_fr=detections['fr'][fin_list[0][0]]
    fname=base+'img1/'+convert(cur_fr)+'.jpg'
    img=cv2.imread(fname)
    print("Writing frames")
    cmd='mkdir '+seqname
    print(cmd)
    os.system(cmd)
    for i in range(len(fin_list)):
 
        new_fr=detections['fr'][fin_list[i][0]]
        if new_fr!=cur_fr:
        
            sname='./'+seqname+'/'+convert(cur_fr)+'.jpg'
            cv2.imwrite(sname,img)
            cur_fr=new_fr
            fname=base+'img1/'+convert(cur_fr)+'.jpg'
            print(fname)
            img=cv2.imread(fname)

        x,y,w,h=detections['x'][fin_list[i][0]],detections['y'][fin_list[i][0]],detections['w'][fin_list[i][0]],detections['h'][fin_list[i][0]]
        img=drawRect(img,x,y,w,h,colors[fin_list[i][1]])
        sname='./'+seqname+'/'+convert(cur_fr)+'.jpg'
    cv2.imwrite(sname,img)
	



    

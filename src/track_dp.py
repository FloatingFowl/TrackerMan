import numpy as np
import cv2, numpy as np, os, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grapher

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
            for node in detections['pr'][i]:
                if detections['dp_c'][node]+detections['r'][i]<mi and detections['free'][node]==-1:
                    mi=detections['dp_c'][node]+detections['r'][i]
                    amin=node

            detections['dp_c'][i]=mi
            detections['dp_link'][i]=amin
            if mi < tot_min:
                tot_min=mi
                tot_amin=i
        print("cost is",tot_min)
        if tot_min>thr_cost:
            break
        final=[]
        cur_ind=tot_amin
        print("Getting path")
        while cur_ind!=-1:
            final.append(cur_ind)
            detections['free'][cur_ind]=1
            cur_ind=detections['dp_link'][cur_ind]
            paths.append(final)
    return paths



def convert(i):
    ti=str(i)
    no=8-len(ti)
    return no*"0" + ti

def drawRect(img,x,y,w,h):
    x1,y1=int(x),int(y)
    x2,y2=int(x+w),int(y+h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    return img




if __name__=="__main__":
    pass
    f=open('pedestrian_detections.pkl','rb')
    dct = pickle.load(f)
    ndct = grapher.graphMaker(dct)
    detections=ndct
    out=dp_many(ndct,10,10,0,0.2,-50,10000000)
    fin_list=[]
    for i in out:
        for j in i:
            fin_list.append(j)
    print("total is",len(fin_list))
    fin_list.sort()

    cur_fr=detections['fr'][fin_list[0]]
    fname='../../seq03-img-left/image_'+convert(cur_fr)+'_0.png'
    img=cv2.imread(fname)
    print("Writing frames")
    for i in range(len(fin_list)):
        if i%100==0:
            print(i)
        new_fr=detections['fr'][fin_list[i]]
        if new_fr!=cur_fr:
        
            sname='./manyres/'+convert(cur_fr)+'.jpg'
            cv2.imwrite(sname,img)
            cur_fr=new_fr
            fname='../../seq03-img-left/image_'+convert(cur_fr)+'_0.png'
            img=cv2.imread(fname)

        x,y,w,h=detections['x'][fin_list[i]],detections['y'][fin_list[i]],detections['w'][fin_list[i]],detections['h'][fin_list[i]]
        img=drawRect(img,x,y,w,h)
        sname='./manyres/'+convert(cur_fr)+'.jpg'
        cv2.imwrite(sname,img)



    

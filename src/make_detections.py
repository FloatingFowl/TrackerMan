import numpy as np
import grapher



def make_detections(filename):
	'''Create the detections object given the file of detections'''
	det_arr=np.loadtxt(filename,delimiter=',')
	detections={'x':[],'y':[],'w':[],'h':[],'fr':[],'r':[]}
	for i in range(len(det_arr)):
		fn=det_arr[i][0]
		xn=det_arr[i][2]
		yn=det_arr[i][3]
		wn=det_arr[i][4]
		hn=det_arr[i][5]
		rn=det_arr[i][6]
		detections['fr'].append(fn)
		detections['x'].append(xn)
		detections['y'].append(yn)
		detections['w'].append(wn)
		detections['h'].append(hn)
		detections['r'].append(rn)

	detections['x']=np.array(detections['x'])
	detections['y']=np.array(detections['y'])
	detections['w']=np.array(detections['w'])
	detections['h']=np.array(detections['h'])
	detections['fr']=np.array(detections['fr'],dtype='uint32')
	detections['r']=np.array(detections['r'])
	return detections

if __name__=="__main__":
	filename='../2DMOT2015/train/PETS09-S2L1/det/det.txt'
	detections=make_detections(filename)
	print("detections done")
	ndct=grapher.graphMaker(detections)
	print('yay')
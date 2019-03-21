import os
import numpy as np
import glob


seqs=glob.glob('../../2DMOT2015/train/*')
for seq in seqs:
	seqname=seq.split('/')[-1]
	imgfolder=seq+'/img1/'
	savefile=seq+'/det/det.pkl'
	cmd='python3 yolo.py --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt '
	cmd+='--image_folder '+imgfolder
	cmd+=' --save_as '+savefile
	print(cmd)
	os.system(cmd)
#cmd='python3 yolo.py --image_folder ../../2DMOT2015/train/TUD-Campus/img1/ --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --save_as TUDcampus.pkl
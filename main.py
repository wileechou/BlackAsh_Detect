# -*- coding: utf-8 -*-
"""
视频背景消除
"""
import os
import argparse
import numpy as np
import cv2
import time
from skimage import data,img_as_float,io
#from moviepy.editor import ImageSequenceClip

def darkChannel(im):
    m, n = np.shape(im)
    patchSize = 3
    padSize = 1
    dark = np.zeros((m, n))
    imj = np.pad(im, padSize, 'constant', constant_values=999)
    for i in range(m):
        for j in range(n):
            patch = imj[i:(i + patchSize ),j:(j+patchSize)]
            dark[i,j] = np.min(patch)

    return dark

def checkout(im):#返回bool，有黑烟为True,无黑烟为False
    dst = img_as_float(im)#255转为0-1
    #print(dst.shape)
    dst = darkChannel(dst)
    #io.imshow(dst)
    #io.show()
    m,n = dst.shape
    white = 0
    black = 0
    for i in range(m):
        for j in range(n):
            if dst[i,j] > 0.99:
                white+=1
            elif dst[i,j] < 0.01:
                black+=1
    return ((m*n-white-black+1)/(m*n))**(-4)

def create_video(photo,video):
    clip = ImageSequenceClip(photo, fps=24)
    clip.write_videofile(video)

def black_ash_car(video_name,out_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    jb = 0
    if not os.path.exists(out_path):
        os.mkdir(out_path)


    cap = cv2.VideoCapture(video_name)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vehicles_cascade = cv2.CascadeClassifier('cardetect.xml')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    test = np.ones([1,300])
    vehilces = []
    #print(test)
    photo=[]
    for i in range(0,num_frames-1,2):
        ret_1,frame_1 = cap.read()
        frame_1 = cv2.resize(frame_1, (600, 400))
        buff_1 = fgbg.apply(frame_1)

        ret_2,frame_2 = cap.read()
        frame_2 = cv2.resize(frame_2, (600, 400))
        buff_2 = fgbg.apply(frame_2)
        #print(i)
        if i>0.2*num_frames:
            #fgmask = buff[:][90:120]

            gray=cv2.cvtColor(frame_1,cv2.COLOR_BGR2GRAY)
            vehilces = vehicles_cascade.detectMultiScale(gray,1.1,2,cv2.CASCADE_SCALE_IMAGE,(50,50))
            for x in vehilces:
                #print(x)
                #write_path2 = '%s/%05d.jpg'%(out_path,jb)
                write_path2 = ''+out_path+'/'+str(i)+'_'+str(jb)+'.jpg'
                bb_1 = cv2.resize(buff_1[min(x[1]+x[3],399):min(x[1]+x[3]*3,400),x[0]:x[0]+x[2]],(30,30))
                bb_2 = cv2.resize(buff_2[min(x[1]+x[3],399):min(x[1]+x[3]*3,400),x[0]:x[0]+x[2]],(30,30))
                # bb里面就是可以处理的图片
                #cv2.imwrite(write_path2,bb)   #  这个是导出图片的函数
                #print(bb.shape)
                #print(checkout(bb))
                if checkout(bb_1) <10 and checkout(bb_2)<10: # func 是判断函数
                    #print(checkout(bb))
                    #cv2.imwrite(write_path2,bb)
                    cv2.rectangle(frame_1, (x[0],x[1]), (x[0]+x[2],x[1]+x[3]), (255, 0, 0),4)
                    cv2.imwrite(write_path2,frame_1)
                    if not os.path.exists(out_path+'/'+str(i)+'.txt'):
                        f=open(out_path+'/'+str(i)+'.txt','w')
                        msg=str(x[0])+','+str(x[1])+','+str(x[2])+','+str(x[3])
                        f.write(msg)
                        f.close()
                    else:
                        f=open(out_path+'/'+str(i)+'.txt','r+')
                        f.read()
                        msg='\n'+str(x[0])+','+str(x[1])+','+str(x[2])+','+str(x[3])
                        f.write(msg)
                        f.close()

                #cv2.rectangle(frame,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(255,0,0),4)
                jb += 1
            cv2.imshow("frame",frame_1)
            #cv2.imshow("frame",buff)
            #print(bb.dtype)

            k = cv2.waitKey(10) & 0xFF
            #photo.append(frame_1)
    #video='demo.mp4'
    #create_video(photo,video)
    cap.release()
    cv2.destroyAllWindows()


black_ash_car('0903-103743.mp4','0903-103743')

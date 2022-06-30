# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np


def ORB_detect(Img, orb):
    
    #Detect features with ORB
    kp = orb.detect(Img,None)
    kp, des = orb.compute(Img, kp)
    
    #Plot the Keypoints in the Original Image. Use Green color
    ORB_img_raw = cv.drawKeypoints(Img, kp, None, color=(0,255,0), flags=0)
    
    return kp, des, ORB_img_raw

def Resize_Img2Plot(img1, img2, img3, factor = 0.5):
    
    resized_img1 = cv.resize(img1,(0,0), None, factor, factor)
    resized_img2 = cv.resize(img2,(0,0), None, factor, factor)
    resized_img3 = cv.resize(img3,(0,0), None, factor, factor)
    
    return resized_img1, resized_img2, resized_img3

def main():
    #Initializations
    cap = cv.VideoCapture(r"C:\Users\Marcos Batista\Documents\git\ComputerVisionProjects\AirCraft_Detection\Dataset\EVision_VideoDataset\1FNB737.avi")
    ret,frame = cap.read()
    last_frame = cv.cvtColor(frame[0:int(0.5*frame.shape[0]),:,:], cv.COLOR_BGR2GRAY)
    ORB_bin = np.zeros(last_frame.shape, dtype = np.uint8)
    Acc_Img = np.zeros(last_frame.shape[0:2], dtype = np.float32)
    last_ORB_bin = ORB_bin[:,:]
    
    #Descriptor initialization
    orb = cv.ORB_create()
    kp = orb.detect(last_frame,None)
    last_kp, last_des = orb.compute(last_frame, kp)


    while(cap.isOpened()):
        #Read image
        ret, frame_raw = cap.read()
        
        if ret == True:
            
            #Crop a ROI and convert to BGR
            frame = frame_raw[0:int(0.5*frame_raw.shape[0]),:,:]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            #cv.imshow('frame', frame)
            
            #Detect keypoints using ORB
            kp, des, ORB_raw = ORB_detect(frame, orb)
            
            #Convert ORB to Binary Image
            ORB_bin = np.zeros(last_frame.shape, dtype = np.uint8)
            for i in range(len(kp)):
                ORB_bin[int(kp[i].pt[1]),int(kp[i].pt[0])] = 255
            #cv.imshow('ORB_Bin', ORB_bin)
            
            #Apply Filter to eliminate punctual kp
            ORB_filter = cv.GaussianBlur(ORB_bin, (7,7), 3)
            #cv.imshow('ORB_Gaussian', ORB_filter)
            
            #Enhance with threshold
            ret, ORB_OTSU = cv.threshold(ORB_filter,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            #cv.imshow('ORB_OTSU', ORB_OTSU)
            
            kernel = np.ones((10,10))      
            ORB_morph = cv.morphologyEx(ORB_OTSU, cv.MORPH_CLOSE, kernel)
              
            
            #Use Weighted Mean Average instead of using last images average
            Acc_Img = cv.accumulateWeighted(ORB_morph, Acc_Img, alpha = 0.8)
            
            #Detect Contours
            contours2,_ = cv.findContours(Acc_Img.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            drawing_acc = cv.drawContours(frame_raw, contours2, -1, (0,255,0), 3)
            
            cv.imshow('Contours_acc', drawing_acc)
            #cv.imshow('Acc_Img', Acc_Img)
    
            
            #Update last detections and frame
            last_frame, last_ORB_bin = frame, ORB_morph
            last_kp, last_des = kp, des
            
            #Use 'q' in Keyboard to stop the video or Space (' ') to run step by step.
            Keyboard = cv.waitKey(1) & 0xFF
            if Keyboard == ord('q'):
                break
            elif Keyboard == ord(' '):
                cv.waitKey(0)
                continue
        else:
            break
        
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
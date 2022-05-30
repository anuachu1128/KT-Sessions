#Importing necessary libraries
import sys
import os
import pandas as pd
import cv2
import mediapipe as mp
import json
import argparse


#1. SUMMARY
def videoSummary(videoPath, saveDirPath):
    summaryDict = {}
    summaryDict['video_path'] = videoPath
    #Getting components of the path seperated by '_' after removing .mp4 extension
    videoPathComp = videoFileName.strip('.mp4').split('_')
    # Extracting necessary parameters and adding it onto the dictionary 
    if(videoPathComp[0][1] == 'H'):
        summaryDict['env']="home"
    else:
        summaryDict['env']="studio"
    summaryDict['signer_id'] = int(videoPathComp[1][1])
    summaryDict['gloss_id'] = int(videoPathComp[2])
    if(videoPathComp[0][1] == 'H'):
        summaryDict['position'] = videoPathComp[3]
    else:
        summaryDict['position'] = "S"
    videoElem = cv2.VideoCapture(videoPath)
    summaryDict['num_of_frames'] = int(videoElem.get(cv2.CAP_PROP_FRAME_COUNT))
    summaryDict['fps'] = float(videoElem.get(cv2.CAP_PROP_FPS))
    summaryDict['height'] = int(videoElem.get(cv2.CAP_PROP_FRAME_HEIGHT))
    summaryDict['width'] = int(videoElem.get(cv2.CAP_PROP_FRAME_WIDTH))
    summaryDict['rgb_path'] = os.path.join(saveDirPath+"/", videoPath.split('/')[-1].strip('.mp4') + '_lowResRGB.avi')
    summaryDict['pose_path'] = os.path.join(saveDirPath+"/", videoPath.split('/')[-1].strip('.mp4') + '_poseEstimate.json')
    return summaryDict

#2. LOW RESOLUTION RGB SQUARE VIDEO
def resizeAndCrop(videoPath, saveDirPath):
    videoElem = cv2.VideoCapture(videoPath.split('/')[-1])
    #Saving the resized and cropped video in the path specified by the user taking in -input video file, codec to compress frames using FourCC code, framerate (fps), resolution
    out = cv2.VideoWriter(os.path.join(saveDirPath+"/", videoPath.split('/')[-1].strip('.mp4') + '_lowResRGB.avi'),cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), float(videoElem.get(cv2.CAP_PROP_FPS)), (320,320))
    while True:
        #Taking each frame of the video
        success, frame = videoElem.read()
        if success == True:
            #Resizing the original frame to 320x320 resolution with no scaling on x (fx) and y (fy) axis using bicubic interpolation
            resizedFrame = cv2.resize(frame,(320,320),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(resizedFrame)
        else:
            break
    print()

#3. BODY AND HAND POSE ESTIMATION
def retrievePoseEstimates(videoPath, saveDirPath):
    videoElem = cv2.VideoCapture(videoPath)
    # Initialize mediapipe Holistic class.
    mpPose = mp.solutions.holistic
    # Setup the Holistic model with 2 parameters: detection confidence of 50% and tracking confidence of 50%
    poseVideo = mpPose.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #Lists to store estimated pose cordinates and hand (both left and right) cordinates
    pose_x = []
    pose_y = []
    right_hand_x = []
    right_hand_y = []
    left_hand_x = []
    left_hand_y = []
    while True:
        success, frame = videoElem.read()
        if success == True:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Make estimation of poses 
            results = poseVideo.process(frameRGB)
            #x and y cordinates of each estimated body landmark i.e., results.pose_landmarks.landmark
            if(results.pose_landmarks):
                #Iterate through each landmark and take its x and y cordinates
                for _,lm in enumerate(results.pose_landmarks.landmark):
                    pose_x.append(lm.x)
                    pose_y.append(lm.y)
            #if pose landmark does not exist- Fill in NaN to both the list
            else:
                pose_x.append('NaN')
                pose_y.append('NaN')
            if(results.right_hand_landmarks):
                for _,lm in enumerate(results.right_hand_landmarks.landmark):
                    right_hand_x.append(lm.x)
                    right_hand_y.append(lm.y)
            else:
                right_hand_x.append('NaN')
                right_hand_y.append('NaN')
            if(results.left_hand_landmarks):
                for _,lm in enumerate(results.left_hand_landmarks.landmark):
                    left_hand_x.append(lm.x)
                    left_hand_y.append(lm.y)
            else:
                left_hand_x.append('NaN')
                left_hand_y.append('NaN')
        else:
            break
    
    with open(os.path.join(saveDirPath+"/", videoPath.split('/')[-1].strip('.mp4') + '_poseEstimate.json'),"w") as outputFile:
        #Adding the pose estimates as a dictionary to the file using dump command to the specified output json file
        json.dump({"pose_x": pose_x, "pose_y": pose_y, "hand1_x": right_hand_x, "hand1_y": right_hand_y, "hand2_x": left_hand_x, "hand2_y": left_hand_y}, outputFile)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    videoPath = args.file_path
    saveDirPath = args.save_dir
    #Checking whether the specified path exist or not
    if(os.path.isfile(videoPath)):
        videoFileName = videoPath.split('/')[-1]
        videoPathComp = videoFileName.strip('.mp4').split('_')
        if(len(videoPathComp)==3 or len(videoPathComp)==4):
            summaryDict = videoSummary(videoPath, saveDirPath)
            print('\n----------------------------------VIDEO SUMMARY CREATED--------------------------------------------\n')
            resizeAndCrop(videoPath, saveDirPath)
            print('\n-------------------------------LOW RESOLUTION RGB VIDEO CREATED--------------------------------------------\n')
            retrievePoseEstimates(videoPath, saveDirPath)
            print('\n--------------------------------BODY POSE ESTIMATES CREATED--------------------------------------------\n')
        else:
            print("\n INVALID!!! NAME OF THE VIDEO FILE IS NOT FOLLOWING THE NAMING CONVENTION\n")
    else:
        print("\nINVALID VIDEO PATH!!!\n")

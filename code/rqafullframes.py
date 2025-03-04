import cv2
import os
import def_functions_rqafullframes

""" ******************************************************************************************************************************************
    In this script: 
    (1) --> there is a script to downscale videos if their resolution is big in order to run faster,
     you can do this for a list of videos or for inly one video
    (2) --> then there is a script to identify the frame change in a list of videos or in a video, and
     the list of frame changes can be stored in a txt file
    (3) --> there is the same script to identify frame changes but this script separates the video in smaller
     parts so the algorithm will run faster, again the result is one list with frame changes
    Depending on the length of yout video script (2) or script (3) can be choosen.
    ****************************************************************************************************************************************** """


# define the path where the videos are stored
path="C:\\Users\\asd\Documents\\BrainSIM\\video\\Autoshot\\annotated_videos"
#path="C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai"
#path="C:\\Users\\asd\Documents\\BrainSIM\\video\\BBC"


"""(1) Downscale a video resolution......................................................................."""
#dscale is helpfull to run rqa full frames faster

#if i have a set of videos that i want to run rqa full frames first i dscale them to run faster
#put all videos that exist in dir "path" in a list, in order to dscale 
#if i only have one video want ti run i make a list with only this one video
os.chdir(path)
l_videos=[]
for i in os.listdir(path):
    if i[-4:]==".mp4":
        l_videos.append(i)

#run dscale for all videos in the list l_videos
n=0
for i in l_videos:
    def_functions_rqafullframes.d_scale(path,i)
    print(n)
    n+=1




"""(2) RUN RQA FULL FRAMES......................................................................................"""
import cv2
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
import json
import time
import def_functions_rqafullframes

file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\Autoshot\\annotated_videos\\Dscale_2"
#file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\BBC\\bbc_dscale2"
os.chdir(file_path)

# # get a list with all the videos in dir to run 
l_videos_dscale=[]
for i in os.listdir(file_path):
    if i[-11:]=="_dscale.mp4":
        l_videos_dscale.append(i)

#if only one video to run make a list contining only that video
#l_videos_dscale=["18592632588_dscale.mp4"]

#RUN FOR ALL VIDEOS IN DIR AND KEEP DICTIONARIES WITH RESULTS
d_frame_change={}
d_sec_change={}
l_time=[]
n=0
time_took=0
for i in  l_videos_dscale:
    vid_name=i[:-11]
    print(vid_name)
    if os.path.exists(file_path+"\\"+vid_name+"_mask5_e15.txt"):    # chose the ending of your files depending of the parameters you want
        print("file exists",n)
    else:
        start_time = time.time()

        frames_flat, fps = def_functions_rqafullframes.load_video_and_flatten(i)     # run funtion to get flatten frames and fps
        rp = def_functions_rqafullframes.psnr_rqa_fullFrames(frames_flat,1,15)       # run function to get rp, D=1, set desired e
        l_f_change, l_s_change =def_functions_rqafullframes.scan_rp(rp,fps)          # run function to scan rp and get frame change, modify the maks for scanning from def_functions_rqafullframes.py


        # save the list l_f_change (cahnges in frames) as txt
        # with open(vid_name+"_mask3_e15.txt", 'w') as f:
        #     f.write(json.dumps(l_f_change))
        # save the list l_f_change (cahnges in sec) as txt
        # with open(vid_name+"_s.txt", 'w') as f:
        #     f.write(json.dumps(l_s_change))

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_took+=elapsed_time
        l_time.append(elapsed_time)
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("Total time until now: ", time_took )
        print(n)
    n+=1




""" (3) FOR LARGE VIDEOS SEPARATE FRAMES IN SMALLER GROUPS TO RUN FASTER......................................................"""
### DEPENDING ON THE LENGTH OF THE VIDEO CHOOSE IN HOW MANY GROUPS TO SEPARATE THE VIDEO IN ORDER TO RUN FASTER
import cv2
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
import json
import time
import def_functions_rqafullframes
#file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai\\rai_dscale4"
file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\BBC\\bbc_dscale2"
#file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\Autoshot\\annotated_videos\\Dscale_2"
os.chdir(file_path)

#get a list with all the videos in dir to run 
l_videos_dscale=[]
for i in os.listdir(file_path):
    if i[-11:]=="_dscale.mp4":
        l_videos_dscale.append(i)

#if only one video to run make a list contining only that video
#l_videos_dscale=["18592632588_dscale.mp4"]

d_frame_change={}
d_sec_change={}
l_time=[]
n=0
time_took=0
for i in  l_videos_dscale:
    vid_name=i[:-11]
    print(vid_name)
    if os.path.exists(file_path+"\\"+vid_name+"_mask5_e18.txt"):
        print("file exists",n)
    else:
        start_time = time.time()

        frames_flat, fps = def_functions_rqafullframes.load_video_and_flatten(i) #run funtion to get flatten frames and fps
        frames_flat_shape = frames_flat.shape
        frames_per_group = frames_flat_shape[0] // 50   #SET THE NUMBER HOW MUCH FRAMES IN EACH GROUP ex. THERE ARE 7500 I WANT 1500 FRAMES PER GROUP SO THE NUM WILL BE 50
        frame_groups = [frames_flat[i:i+frames_per_group] for i in range(0, len(frames_flat), frames_per_group)]

        print(frame_groups)
        l_frame_change=[]
        n1=0
        for group in frame_groups:
            print(vid_name, "group ", n1)
            rp = def_functions_rqafullframes.psnr_rqa_fullFrames(group,1,18)             #run function to get rp, D=1 set the value of e
            l_f_change, l_s_change =def_functions_rqafullframes.scan_rp(rp,fps)          #run function to scan rp and get frame change
            l_f_change_realfnumbers=[item + n1*frames_per_group for item in l_f_change]
            #print(l_f_change_realfnumbers)
            for i in l_f_change_realfnumbers:
                l_frame_change.append(i)
            n1+=1
            #print(l_frame_change)

        #save the list l_frame change as txt
        with open(vid_name+"_mask5_e18.txt", 'w') as f:
            f.write(json.dumps(l_frame_change))

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_took+=elapsed_time
        l_time.append(elapsed_time)
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("Total time until now: ", time_took )
        print(n)
    n+=1


print(l_frame_change)









     
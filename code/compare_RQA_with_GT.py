import os
import pickle
import json
import numpy as np
import def_functions_rqafullframes


#### COMPARE RQA RESULTS WITH GROUND TRUTH RESULTS (GT or gt)............................................................................................

# navigate to the directory where the GT results are stored
file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai"
# file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\BBC\\BBC_Planet_Earth_Dataset\\annotations\\shots"
os.chdir(file_path)
# get all the names of the files for the gt results for all the videos in one list
l_videos_gt=[]
for i in os.listdir(file_path):              # chose the correct ending for your GT files
    if i[-6:]=="gt.txt":
    #if i[-3:]=="txt":
        l_videos_gt.append(i)

# navigate to the directory where the RQA results are stored
file_path2 = "C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai\\rai_dscale4"
# file_path2 = "C:\\Users\\asd\Documents\\BrainSIM\\video\\BBC\\bbc_dscale2"
os.chdir(file_path2)
# get all the names of the files for the gt results for all the videos in one list
l_videos_rqa=[]
for i in os.listdir(file_path2):             # choose rqa results based on their parameters and name of the files
    if i[-9:]=="f_e15.txt":                  # for e18 mask3
    #if i[-9:]=="5_e18.txt":                 # for e18 and mask5
        l_videos_rqa.append(i)

# if you can to compare only one video make the list l_videos_gt with one item 
# the name of the file with GT for that video and the same for RQA
# file_path = "C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai"
# file_path2 = "C:\\Users\\asd\Documents\\BrainSIM\\video\\video_rai\\rai_dscale4"
# l_videos_gt=["21829_gt.txt"]
# l_videos_rqa=["21829_f_e15.txt"]


# create an empty dictioray where the comparison results will be added
d={"video":[],"scene_matches":[],"instant_matches":[],"trans_matches":[],"extra_scenes":[],"missed_scenes":[],"missed_transactions":[]}
n1=0
# loop for all the videos in your lists
for i in range(len(l_videos_gt)):
    vid_name=l_videos_gt[i][:-7]                            # get video name, -7 can be different in different files
    with open(file_path2+"\\"+l_videos_rqa[i], "r") as f:   # open rqa results in a list
        l_rqa = json.loads(f.read())
    gt = np.loadtxt(file_path+"\\"+ l_videos_gt[i])         # open GT result in an np array

    d_matches,l_extra,l_missed,l_missed_t,l_match_ins,l_match_tr=def_functions_rqafullframes.find_compare_scenes(gt,l_rqa)  # run finction fro comparison

    # store comparison results in the dictionary
    d["video"].append(vid_name)
    d["scene_matches"].append(len(d_matches))
    d["instant_matches"].append(len(l_match_ins))
    d["trans_matches"].append(len(l_match_tr))
    d["extra_scenes"].append(len(l_extra))
    d["missed_scenes"].append(len(l_missed))
    d["missed_transactions"].append(len(l_missed_t))

    n1+=1

# chose the directory you want to save the resulting dataframe
os.chdir(file_path2)
import pandas as pd
df = pd.DataFrame(d)  # make the dictionary a dataframe
# save df as xlsx, choose the name of the file
df.to_excel("compare_BBC_gt_rqa_e18_mask5_res.xlsx", index=False) 




### CHANGE THE FORMAT OF COMPARISON RESUTLTS FRAMES --> TIME..................................
# get results of a video in different format from frames to seconds or minutes so you can check on the video
print(l_extra)
np.array(l_extra)/25  # here 25 is fps, change it according to fps of ypur video
print(l_missed)
np.array(l_missed)/25 # fps=25
sec=np.array(l_missed_t)/25 # fps=25
minutes_with_seconds = sec // 60  # Get whole minutes
remaining_seconds = sec % 60      # Get remaining seconds
minutes_with_seconds_str = [f"{m}m {s}s" for m, s in zip(minutes_with_seconds, np.round(remaining_seconds,2))]
print(minutes_with_seconds_str)






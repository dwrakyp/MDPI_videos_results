import os
import cv2


""" *********************************************************************************************************************************************************
    Here we define functions used for rqa in full frames for scene change detection
    d_scale                --> downscales the input video so our algorithm will run faster
    load_video_and_flatten --> loads video, converts to grayscale and flatten pixel values to arrays like (number of frames, flatten pixel values)
    psnr_rqa_fullFrames    --> runs rqa in full frames without embending and gets the RP
    scan_rp                --> scans the RP with specific masks to define where there is a scene change according to the RP
    find_compare_scenes    --> compares the results from rqa with the GT results
   **********************************************************************************************************************************************************"""

"""d_scale........................................................................................................................."""
#FUNTION TO DSCALE A VIDEO IN HALF DIMENTIONS (/2) ................................................................................
#if i want i can dscale more the video changing the output width and height
def d_scale(path,input_video):
    os.chdir(path)
    # Load the input video
    video_capture = cv2.VideoCapture(input_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.chdir(path+"\\dscale2")                  # make sure first that the dir dscale2 exists    
    # Define the output video parameters
    output_width = width // 2                   # New width (half of original)
    output_height = height // 2                 # New height (half of original)
    output_video_path = input_video[:-4] + "_dscale2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height), isColor=True)

    
    # Read frames from the input video, resize, and write to the output video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize the frame to the desired dimensions
        resized_frame = cv2.resize(frame, (output_width, output_height))
        
        # Write the resized frame to the output video
        output_video.write(resized_frame)

    # Release video objects
    video_capture.release()
    output_video.release()
    print("Downscaling complete. Output video saved as:", output_video_path)
    return



"""load_video_and_flatten..................................................................................."""
###RUN RQA FULL FRAMES......................................................................................
# Τhis part cosists of 3 functions load_video_and_flatten, psnr_rqa_fullFrames and scan_rp
import cv2
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
import json


#FUNTION TO LOAD THE INPUT VIDEO CONVERT TO GRAY SCALE AND FLATTEN PIXXELS
def load_video_and_flatten(in_vid):
    video_capture=cv2.VideoCapture(in_vid)
    ret, first_frame = video_capture.read()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    ## i want to convert the 3 color chanels to grayscale
    ## make an empty list to store the frames
    frames_o = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Append the grayscale frame to the frames list
        frames_o.append(gray_frame)
    # Release video object
    video_capture.release()
    #continue with whole video
    frames=frames_o

    ### make each frame flat and then add it to a list
    ### frame_flatten (number of frames, flatten pixel values)
    frames_flatten=[]
    for i in frames:
        flat_frame=i.flatten()
        frames_flatten.append(flat_frame)
    frames_flatten = np.array(frames_flatten)
    return frames_flatten,fps



"""psnr_rqa_fullFrames.................................................................................."""
#RQA function
def psnr_rqa_fullFrames(v0,D,e):      #  v0=(number of frames, flatten patches),D embedding, e threshold
    
    sq_dist_matrix = squareform(pdist(v0, metric='sqeuclidean'))   #compute all pairwise distances from v0, store them in a matrix
    MN=D*v0.shape[1]
    mask=np.eye(D)    #create a mask of size DxD with 1 only in diagonals
    sq_dist_rqa1 = convolve2d(sq_dist_matrix, mask, mode='valid')    #convolve2d slides mask on sq_dist_matrix summing all diagonal element in the mask 
                                                                     #resulting a matrix with n-(D-1) size, this is the matrix prepared for RQA with euclidean distances
    mean_sqerror1 = sq_dist_rqa1/MN
    psnr_matrix1 = 10*np.log10(255**2/mean_sqerror1)   # convert the eu. dist matrix to psnr matrix
    rq_plot = np.where(psnr_matrix1 >= e, 1, 0)        # compute the RP with threshold e

    #RQ plot
    # vmin, vmax = 0, 1
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # plt.imshow(rq_plot, cmap='binary', norm=norm, origin='lower')
    # plt.text(3, 7, 'e_psnr='+str(e), fontsize=12, color='red')
    # plt.title("Recurrence Plot for frames")
    # plt.xlabel("Frames",fontsize=14)
    # plt.ylabel("Frames",fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.show()
    return rq_plot




"""scan_rp..................................................................................."""
### FUNCTION FIND THE FRAME POSITIONS WHERE THERE IS A SCENE CHANGE FROM RP
def scan_rp(rp,fps):
    # Create the mask of [0 1] that will scan the RP
    # mask1 = np.array([[1, 0, 0],      #mask2
    #                 [0, 1, 1],
    #                 [0, 1, 1]])  

    # mask1 = np.array([[1, 0, 0, 0],       #mask3
    #                 [0, 1, 1, 1],
    #                 [0, 1, 1, 1],
    #                 [0, 1, 1, 1]])
    
    mask1 = np.array([[1, 0, 0, 0, 0, 0],     #mask5
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1]])    

    # mask1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],     #mask7
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1]])   

    # mask1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],     #mask9
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])                  


    mask_rows1, mask_cols1 = mask1.shape

    #find all the statring possitions of scenes that cover the RP diagonal with black a lot
    l_scenestart_pos = []
    # Iterate over each position in the larger matrix
    for i in range(rp.shape[0] - mask_rows1 + 1):
        for j in range(rp.shape[1] - mask_cols1 + 1):
            if i == j:  # Only apply the mask along the main diagonal
                # Extract the submatrix from the larger matrix
                submatrix = rp[i:i+mask_rows1, j:j+mask_cols1]
                # Check if the submatrix matches the mask
                if np.array_equal(submatrix, mask1):
                    if j in l_scenestart_pos:
                        pass
                    else:
                        l_scenestart_pos.append(j)

    #find all ending positions of a scene that cover the RP diagonal a lot so i can find the starting possition of scene that donot cover a lot the diagonal
    l_sceneend_pos=[]

    # mask2 = np.array([[1, 1, 0],        #mask2
    #                 [1, 1, 0],
    #                 [0, 0, 1]])

    # mask2 = np.array([[1, 1, 1, 0],       #mask3
    #                 [1, 1, 1, 0],
    #                 [1, 1, 1, 0],
    #                 [0, 0, 0, 1]])

    mask2 = np.array([[1, 1, 1, 1, 1, 0],     #mask5
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

    # mask2 = np.array([[1 ,1, 1, 1, 1, 1, 1, 0],     #mask7
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1, 1, 1, 1, 1, 1, 1, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 1]])
    
    # mask2 = np.array([[1 ,1, 1 ,1, 1, 1, 1, 1, 1, 0],     #mask9
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [1 ,1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                 [0 ,0, 0, 0, 0, 0, 0, 0, 0, 1]])


    mask_rows2, mask_cols2 = mask2.shape
    for i in range(rp.shape[0] - mask_rows2 + 1):               # Iterate over each position in the larger matrix
        for j in range(rp.shape[1] - mask_cols2 + 1):
            if i == j:                                          # Only apply the mask along the main diagonal
                submatrix = rp[i:i+mask_rows2, j:j+mask_cols2]  # Extract the submatrix from the larger matrix
                if np.array_equal(submatrix, mask2):            # Check if the submatrix matches the mask
                    if j+3 in l_sceneend_pos:
                        pass
                    else:
                        l_sceneend_pos.append(j+3)


    # select only the frame change position once, create a list with all frames where there ir a change
    # each item of the list l_frame_change indicates the ending of the scene and item+1 is the start of the new scene
    # the number of frame i will add in l_frame_change as a starting possition will be at least 10 frames away from the previous starting frame position, changes between 10 frames does not make sence to fast for the eye or maybe the algorthim detect the same change twice
    l_frame_change=[]
    for i in l_scenestart_pos:                                  #loop through l_scnestart_pos
        if any(i - j in l_frame_change for j in range(0, 11)):  #check if any of the 10 number before i exists in the l_frame_change
            pass                                                #if any exist then pass
        else:
            l_frame_change.append(i+1)                          #if it does not exist then append i+1 in the list
    # the same for scene ending posistion i check if there is an ending position 10 frame further from my possition
    for i in l_sceneend_pos:
        if any(i + j in l_frame_change for j in range(0, 11)):
            pass
        else:
            l_frame_change.append(i+1)

    l_frame_change.sort()               #sort the frame changes in the list


    l_framechange_sec=np.round(np.array(l_frame_change) / fps, 1)
    return l_frame_change,l_framechange_sec


"""find_compare_scens..................................................................................................................."""
###.....................................................................................................................................
#### TO COMPARE RQA RESULTS OF SCENE CHANGE WITH GROUND THROUTH ANNOTATION OF VIDEOS....................................................
def find_compare_scenes(gt,l_rqa):
    # GT result are stored in different format in np arrays containing starting and endind positon for each scene 
    # ex. [0  15]
    #     [16 34]
    #     [35 57]
    # we want to convert this type of result to a list and keep only the ending frames of scenes
    # and keep in separate lists staring and ending of the transition frames 
    # so we can compare GT with our results
    #create two list with gt start and gt end
    gt_s = [row[0] for row in gt]  
    gt_e = [row[1] for row in gt] 

    # add all frames in one list, so it will be comparable to out result of frame changes which is in one list
    gt_f=[]
    for i in range(len(gt_e)):
        gt_f.append(gt_s[i])
        gt_f.append(gt_e[i])

    # remove consecutive frames that are the ending of one scene and the starting of an other
    # keep only the ending of the scenes, like in the list i get from rqafullframes.py
    i = 0
    while i < len(gt_f) - 1:
        if gt_f[i] + 1 == gt_f[i + 1]:
            del gt_f[i + 1] 
        else:
            i += 1
    gt_f=gt_f[1:-1]


    # find transitions frames of gt and put start info frame in a list and ending frame in an other one
    # transition frames are the frame where the scene change is not instant but gradual and last for some frames
    trans_frames_st=[]
    trans_frames_end=[]
    for i in range(gt.shape[0]-1):  
        if abs(gt[i][1]-gt[i+1][0])>1:
            trans_frames_st.append(gt[i][1])
            trans_frames_end.append(gt[i+1][0])

    l_rqa_F=l_rqa.copy()
    gt_F=gt_f.copy()

    for j in trans_frames_end:      # remove all transition endings from the gt_F
        gt_F.remove(j)

    # for transitions rqa might capture more than one frames in tha range of the transition, so for every transition that rqa captures
    # keep only one frame in the final list l_rqa_F if the algorithm captures more
    for i in range(len(trans_frames_st)):   
        l_in_trans_lrqa = [num for num in l_rqa if trans_frames_st[i] <= num <= trans_frames_end[i]] # make a list of numbers that are in l_rqa and are found between gradual transition in GT
        if l_in_trans_lrqa:                                                                          # if the list in not empty keep onlt the first item of the l_in_trans_lrqa in l_rqa_F
            for ii in range(len(l_in_trans_lrqa)):                                                  
                if l_in_trans_lrqa[ii] in l_rqa_F:
                    l_rqa_F.remove(l_in_trans_lrqa[ii])
            l_rqa_F.append(trans_frames_st[i])

    # sort the final list for comparison
    l_rqa_F.sort()
    gt_F.sort()

    # compare the 2 lists l_rqa_F and gt_F aling their items, a match is considered if two items have smaller distance than 5 and smaller distance than the minimun of
    # of the next closer item in the two lists--> min(next_diff1,next_diff2), missing or extra items are asinged correctly to missed or extra scenes
    d_scene_matches = {} #d={scenerqa:scenegt}
    l_extra_scenes = []
    l_missed_scenes = []
    i1 = 0
    i2 = 0
    while i1 < len(l_rqa_F) and i2 < len(gt_F):
        item1 = l_rqa_F[i1]
        item2 = gt_F[i2]
        
        diff1 = abs(item1 - item2)                     # find the diff between the items of the lists
        
        if i1 < len(l_rqa_F) - 1:                      # for l_rqa
            next_diff1 = abs(l_rqa_F[i1 + 1] - item2)  # find the difference between i1+1 next item
        else:
            next_diff1 = float('inf')                  # when there is no next item set diff to inf
        
        if i2 < len(gt_F) - 1:                         # the same for gt_f
            next_diff2 = abs(item1 - gt_F[i2 + 1])
        else:
            next_diff2 = float('inf')
        
        if diff1<=5 and diff1 <= min(next_diff1, next_diff2):     # if diff is smaller than min(next_diff1, next_diff2) and smaller than 5 consider it a match
            d_scene_matches[item1] = item2
            i1 += 1
            i2 += 1
        else:                                                     # else  find in witch list there is an extra element and append in the correst list
            if next_diff1 < next_diff2:
                l_extra_scenes.append(l_rqa_F[i1])
                i1 += 1
            else:
                l_missed_scenes.append(gt_F[i2])
                i2 += 1
    while i2 < len(gt_F):                                         # Handle remaining elements of gt_f
        if gt_F[i2] in d_scene_matches:
            pass
        else:
            l_missed_scenes.append(gt_F[i2])
        i2 += 1

    # from the items in missed scenes keep in a list those that a found in fradual transition   
    l_missed_trans=[]
    n=0
    for i in l_missed_scenes:
        for j in range(len(trans_frames_st)):
            if trans_frames_st[j] <= i <= trans_frames_end[j]:
                if i in l_missed_trans:
                    pass
                else:
                    l_missed_trans.append(i)
   
    #create two list one with all the instane scene change matches and one with all the transition matches
    l_match_inst=[]
    l_match_trans=[]   
    for i in d_scene_matches:
        for j in range(len(trans_frames_st)):
            if trans_frames_st[j] <= i <= trans_frames_end[j]:
                if i in l_match_trans:
                    pass
                else:
                    l_match_trans.append(i)
        if i not in l_match_trans:
            l_match_inst.append(i)


    print("\extra scenes:")
    print(l_extra_scenes)
    print("\nmissed scenes:")
    print(l_missed_scenes)
    print("\missed transactions:")
    print(l_missed_trans)
    #d_scene_matches --> dictionary with all the scene match {frame from rqa : frame from gt}
    #l_extra_scene   --> all the frames that are found from rqa as scene change but do not exist in gt
    #l_missed_scene  --> all the frames that exist in gt but are not founr with rqa
    #l_missed_trans  --> all the transition that exist in gt but are not found with rqa
    #l_match_inst    --> all the frames from instant changes that exist both in gt and rqa
    #l_match_trans   --> all the frames fron transitions that exist both in gt and rqa
    return d_scene_matches,l_extra_scenes,l_missed_scenes,l_missed_trans,l_match_inst,l_match_trans




""" ********************************************************************************************************************************************************************
    In this script all the functions used for RQA PSNR are defined, for motion detection in the pathces of the frames:
     patches_unique       --> to create a grid of non overlapping patches in the frames and store their flatten values
     export_grayscale_mp4 --> to save the video in grayscale mp4
     psnr_rqa_rp          --> run rqa based on psnr values and get the RP
     pairwise_psnr        --> calculate paiwise psnr form one patch and a spesific D, used for fnn for optimal D (embending dimention)
     psnr_rqa1            --> to run rqa based on psnr values and compute rqa metrics REC,DET,Lavg_lmin,ent,LAM,TT4
    ******************************************************************************************************************************************************************** """


import imageio
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
import matplotlib.pyplot as plt


"""patches_unique ................................................................................................................................................................"""
#### CREATE A GRID OF PATCHES IN THE FRAMES.......................................................................................................................................
# define a function that makes an array of flatten unique pathces the next
# patch start after the end of the previous one in horizontal and vertical axes
def patches_unique(patch_size,image):
    image_size = image.shape
    patches_list = []
    # compute the number of patches horizonticaly and vetricaly
    if image_size[0] % patch_size == 0:
        no_patches1 = image_size[0] // patch_size
    else:
        no_patches1 = image_size[0] // patch_size + 1
    if image_size[1] % patch_size == 0:
        no_patches2 = image_size[1] // patch_size
    else:
        no_patches2 = image_size[1] // patch_size + 1
    # create patches and save them in a list
    for i in range(0,no_patches1):
        for j in range(0,no_patches2):
            p = image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
            p_flat = p.flatten()
            # for the last patch that might have smallew size i append
            # zeros to the end of the flatten patch so it will have the
            # size as the rest of the pathces
            if p_flat.size < patch_size*patch_size:
                desire_size=patch_size*patch_size
                zeros=desire_size-p_flat.size
                zeros_array = np.zeros(zeros)
                p_flat = np.concatenate((p_flat, zeros_array))
            patches_list.append(p_flat)
    p_list=np.array(patches_list)
    return p_list,no_patches1,no_patches2


"""export_grayscale_mp4......................................................................................................................"""
### FUNTION TO SAVE THE DIVEO IN GRAYSCALE...................................................................................................
import imageio
def export_grayscale_mp4(frames, output_path, fps):
    # Create a writer for the MP4 video
    writer = imageio.get_writer(output_path, fps=fps)
    # Iterate through frames and write to the video
    for frame in frames:
        # Ensure the frame is of type uint8
        frame = np.asarray(frame, dtype=np.uint8)
        # Write the frame to the video
        writer.append_data(frame)
    # Close the writer
    writer.close()

"""psnr_rqa_rp............................................................................................................................."""
## RQA.....................................................................................................................................
### TO RUN RQA BASED ON PSNR VALUES AND COMPUTE THE RP
def psnr_rqa_rp(v0,D,e,p):      # for a patch v0=(number of frames, flatten patches),D embedding, e threshold, p patch, lmin minimum diagonal lenth for metrics
    
    sq_dist_matrix = squareform(pdist(v0, metric='sqeuclidean'))   #comPute all pairwise distances form v0, store them in a matrix
    MN=D*v0.shape[1]
    #print("MN",MN)
    mask=np.eye(D)    #create a mask of size DxD with 1 only in diagonals
    sq_dist_rqa1 = convolve2d(sq_dist_matrix, mask, mode='valid')    #convolve2d slides mask on sq_dist_matrix summing all diagonal element in the mask 
                                                                     #resulting a matrix with n-(D-1) size, this is the matrix prepared for RQA with euclidean distances
    mean_sqerror1 = sq_dist_rqa1/MN
    psnr_matrix1 = 10*np.log10(255**2/mean_sqerror1)   # convert the eu. dist matrix to psnr matrix
    rq_plot = np.where(psnr_matrix1 >= e, 1, 0)        # compute the RP with threshold e

    #RQ plot
    vmin, vmax = 0, 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(rq_plot, cmap='binary', norm=norm, origin='lower')
    #plt.text(3, 7, 'e_psnr='+str(e), fontsize=12, color='red')
    plt.title("Recurrence Plot, patch No "+str(p),fontsize=16)
    plt.xlabel("Frames",fontsize=14)
    plt.ylabel("Frames",fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return rq_plot



"""pairwise_psnr..................................................................................................................................................................."""
## FNN (FALSE NEAREST NEIGHBORS) FOR EMBENDING DIMENSION PSNR (D)..................................................................................................................
#calculate paiwise psnr form one patch and a spesific D
def pairwise_psnr(patch,D):
    tau=1                             # rqa parameter
    Vsize=patch.shape[0]-(D-1)*tau    # Rqa V vector size influenced by D
    v0=[]
    for i in range(0,Vsize):          #  to create rqa vector V 
        Vx = patch[i:i+D]
        v0.append(Vx)
    v0 = np.array(v0)                 

    v_for_msqer=[]                    # make each matrix flat V--> v_for_msqer (here)
    for i in range(v0.shape[0]):
        vx_flat=v0[i].flatten()
        v_for_msqer.append(vx_flat)
    v_for_msqer=np.array(v_for_msqer)
    MN=v0.shape[1]*v0.shape[2]        # total number of elements in each patch
    #print("MN",MN)

    #compute pasn matrix
    sq_error = squareform(pdist(v_for_msqer, metric='sqeuclidean'))   
    mean_sqerror = sq_error/MN
    pairwise_psnr = 10*np.log10(255**2/mean_sqerror)
    return pairwise_psnr



"""psnr_rqa1........................................................................................................"""
### RQA WITH METRICS FOR EACH PATCH WITH THE DOPT...................................................................

from scipy.signal import convolve2d
from collections import Counter
def psnr_rqa1(v0,D,e,p,lmin,umin):      # for a patch v0=(number of frames, flatten patches),D embedding, e threshold, p patch, lmin minimum diagonal lenth for metrics
    
    sq_dist_matrix = squareform(pdist(v0, metric='sqeuclidean'))   #comoute all pairwise distances form v0, store them in a matrix
    MN=D*v0.shape[1]
    mask=np.eye(D)    #create a mask of size DxD with 1 only in diagonals
    sq_dist_rqa1 = convolve2d(sq_dist_matrix, mask, mode='valid')    #convolve2d slides mask on sq_dist_matrix summing all diagonal element in the mask 
                                                                     #resulting a matrix with n-(D-1) size, this is the matrix prepared for RQA with euclidean distances
    mean_sqerror1 = sq_dist_rqa1/MN
    psnr_matrix1 = 10*np.log10(255**2/mean_sqerror1)   # convert the eu. dist matrix to psnr matrix
    rq_plot = np.where(psnr_matrix1 >= e, 1, 0)        # compute the RP with threshold e
    
    # RQA metrics        
    total_points = rq_plot.size                        
    recurrent_points = np.sum(rq_plot)
    REC = round((recurrent_points / total_points),3)  #Reccurence Rate
    #diagonals
    num_diagonals=[]
    for i in range(rq_plot.shape[0]):   #to count the all the diafonals
        for j in range(rq_plot.shape[1]):
            if rq_plot[i, j] == 1 and (0 in (i, j) or rq_plot[i-1, j-1] == 0):
                line_length = 0
                while (i + line_length < rq_plot.shape[0] and j + line_length < rq_plot.shape[1]):
                    if rq_plot[i + line_length, j + line_length] == 1:
                        line_length += 1 
                    elif rq_plot[i + line_length, j + line_length] == 0:
                        break
                if line_length > 0:
                    num_diagonals.append(line_length)
    total_diagonals = len(num_diagonals)
    
    length_counts = Counter(num_diagonals)
    #print(length_counts)
    #all rec point without the LOI
    sum_l_all = sum(i * count for i, count in length_counts.items() if i< rq_plot.shape[0])
    sum_sketo = sum(length_counts.values())
    sum_l_min = sum(i * count for i, count in length_counts.items() if i > lmin and i < rq_plot.shape[0])
                                                                               #this here to exclude the main diagonal        
    sum_lmin_hist = sum(count for i, count in length_counts.items() if i > lmin and i < rq_plot.shape[0])
    #caclulate DET without LOI
    if sum_l_all == 0:
        DET = 0 #or NaN                     # Determism
    else:
        DET = round(((sum_l_min)/(sum_l_all)),3)

    #Lavg for diagonal>lmin
    if sum_lmin_hist == 0:
        Lavg_lmin =0   #or NaN
    else:
        Lavg_lmin=round(sum_l_min/sum_lmin_hist,3)
    
    if len(list(length_counts)) == 1:
        Lmax=0
    else:
        Lmax = sorted(list(length_counts))[-2]                                            #this here to exclude the main diagonal                                  
    d_prob_dist = {i: count / sum_lmin_hist for i, count in length_counts.items() if i > lmin and i < rq_plot.shape[0]}
    #print(d_prob_dist)
    ENT = -sum(prob * np.log2(prob) for prob in d_prob_dist.values())   #Entropy
    ent=round(ENT,3)
    #print(ent,d_prob_dist)

    ### to plot histogram
    # lengths = list(d_prob_dist.keys())
    # probabilities = list(d_prob_dist.values())
    # plt.bar(lengths, probabilities, color='blue', alpha=0.7)
    # plt.xlabel('Diagonals length')
    # plt.ylabel('prob')
    # plt.title('Histogram of Probability Distributions')
    # plt.grid(True)

    #verticals
    num_verticals = []
    recurrent_points = np.sum(rq_plot)    
    for i in range(rq_plot.shape[0]):     #to count all the verticals
        for j in range(rq_plot.shape[1]):
            if rq_plot[i, j] == 1:
                if rq_plot[i, j] == 1 and (j==0 or rq_plot[i, j-1] == 0):
                    line_length = 0
                    while (i < rq_plot.shape[0] and j+1 < rq_plot.shape[1]+1 and j+line_length < rq_plot.shape[1]):
                        if rq_plot[i , j + line_length] == 1:
                            line_length += 1 
                        else:
                            break
                    if line_length > 0:
                        num_verticals.append(line_length)
    length_counts = Counter(num_verticals) 
    sum_l_all = sum(i * count for i, count in length_counts.items()) #all rec points
    sum_ver_over1 = sum(i*counts for i, counts in length_counts.items() if i>1) #all rec points in ver over 1 length
    #sum_sketo = sum(length_counts.values()) # only if i want to calulate Vavg in general
    sum_l_min = sum(i * count for i, count in length_counts.items() if i > umin)
    sum_lmin_hist = sum(count for i, count in length_counts.items() if i > umin)
    if sum_l_all == 0:
        LAM = 0     #or NaN   #Laminarity
    else:
        LAM=round(sum_ver_over1/sum_l_all,3) #rec in ver/ all rec point
    if sum_lmin_hist == 0:
        TT =0    #or NaN    #Traping Time
    else:
        TT=round(sum_l_min/sum_lmin_hist,3)  # points in vert of umin/ number of ver lines over umin
    Vmax=max(length_counts)
    l_res=[REC,DET,Lavg_lmin,ent,LAM,TT]  # Lmax,Vmax


    # #RQ plot
    # vmin, vmax = 0, 1
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # plt.imshow(rq_plot, cmap='binary', norm=norm, origin='lower')
    # #plt.text(3, 7, 'e_psnr='+str(e), fontsize=12, color='red')
    # plt.title("Recurrence Plot, patch No "+str(p),fontsize=16)
    # plt.xlabel("Frames",fontsize=14)
    # plt.ylabel("Frames",fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.show()
    return l_res



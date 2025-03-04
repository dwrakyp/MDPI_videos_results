import cv2
import numpy as np
import def_functions_rqaPSNR
import os
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors
import matplotlib.pyplot as plt

""" ******************************************************************************************************************************************************************************
    In this script:
    (1) --> open video convert to grayscale, make grid of patches and create array for rqa with flatten pixel values (number of patches, frames, elements in each path)
    (2) --> run FNN for optimun embending dimension (Dopt) for each patch, and plots: histogram of Dopt and heatmap with greyscale values of Dopt
    (3) --> rqa with metric for the Dopt of each patch and plots : histogram of RR values in patches, scatter plot RR vs Dopt, and heatmap with greyscale values of RR
    (4) --> rearrange the frames of the video accorind to Dopt or RR and compress with H264 to visualise the rearranged frames

    ******************************************************************************************************************************************************************************"""


"""(1) .............................................................................................................................................................."""
#### navigate to direction of the videos
os.chdir("C:\\Users\\asd\Documents\\BrainSIM\\video\\UCF101_videos\\UCF101")

vid_name="v_BaseballPitch_g01_c03.avi"
video_capture=cv2.VideoCapture(vid_name)

## ret is a boolean T if there is a frame F if there is not
## the size of the frames is (240,352,3) meaning 240 pixels in vertical
## 352 pixels in horizontal and 3 color chanels
ret, first_frame = video_capture.read()

## i want to convert the 3 color chanels to grayscale
## make an empty list to store the frames
frames = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Append the grayscale frame to the frames list
    frames.append(gray_frame)

# Convert the list of frames to a NumPy array
video_array = np.array(frames)
# Release video object
video_capture.release()

Fr=frames

#  to save video as mp4
# import imageio
# def_functions.export_grayscale_mp4(Fr,"g_" + vid_name+".mp4",25)


# go to main directory again
os.chdir("C:\\Users\\asd\\Documents\\BrainSIM")
# plot a frame of the video in grayscale
import matplotlib.pyplot as plt
plt.imshow(frames[40], cmap='gray')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()



patch_size=16    # set the patch size ex. 16x16
## UNIQUE PATHCES
## create patches in each one of the frames, determine the patch size
## store the in an array (frames, number of patches , elements in each path)
l_frames_allpatches=[]
for i in range(len(Fr)):
    p=def_functions_rqaPSNR.patches_unique(patch_size,Fr[i])[0] #chose the size of the patches
    l_frames_allpatches.append(p)
ar_frames_allpatches = np.array(l_frames_allpatches)
## re arrange the array in order to have an array containing matrices of
## the patch elements in each frame in the same patch position to run rqa
##  the new array (number of patches, frames, elements in each path)
ar_pre_rqa = ar_frames_allpatches.transpose(1, 0, 2)



""" (2) FNN for embending dimension psnr.........................................................................................................................."""
## FIND Dopt FOR EACH PATCH
import time                
start_time = time.time()     # compute the time it takes to run

l_D=list(range(1,10,1))      # choose a list of embeding dimensions
l_p_zeros_l_psnr=[]
l_p_zeors_h_psnr=[]
l_opt=[]
d_patch_Dopt={}
for p in range(0,ar_pre_rqa.shape[0]): # run for all patches
    print(p)                             
    p0=ar_pre_rqa[p]
    l_per=[]
    l_mean_maxpsnr=[]
    for i in range(len(l_D)-1):       # loop for all D 
        fnn_counts=0                  # initalize FNN counts
        counts=0
        l_psnr_maxval=[]

        psnr_D_i = def_functions_rqaPSNR.pairwise_psnr(p0,l_D[i])          # compute psnr matrix for D
        psnr_D_1plusi = def_functions_rqaPSNR.pairwise_psnr(p0,l_D[i+1])   # compute psnr matrix for D+1
        for ii in range(psnr_D_1plusi.shape[0]):     # loop for all embeded vectors
            counts+=1
            val_max_i = sorted(psnr_D_i[ii,:psnr_D_1plusi.shape[0]])[-2]                 # find max for each vector
            max_index = np.where(psnr_D_i[ii,:psnr_D_1plusi.shape[0]]==val_max_i)[0][0]  # keep the index of max value
            val_index_plus1D = psnr_D_1plusi[ii,:][max_index]                            # find value at the same index for D+1

            l_psnr_maxval.append(val_max_i)

            if  abs(val_max_i-val_index_plus1D)>5:       # find the maximun psnr value between 2 vectos v for
                fnn_counts+=1                            # D and D+1 and compare them if their difference is above a threshold of FNN, here 5 db
            per_fnn=fnn_counts/counts                    # if yes consider them as false neibhors if the 
                                                         # count fnn percenatge for all vectors
        l_per.append(per_fnn)
        l_mean_maxpsnr.append(np.mean(l_psnr_maxval))
        #find the mean values of the array of psnr for D
        rows, cols = psnr_D_i.shape
        mask = ~np.eye(rows, cols, dtype=bool) # Create a mask to exclude the diagonal elements
        mean_psnr_D = np.mean(psnr_D_i[mask])

    # find optimum D acording to the critiria mentioned    
    if l_per[0]==0 or l_per[0]<0.2:
        if mean_psnr_D>35:
            D_opt=25
            l_p_zeors_h_psnr.append(p)
        else:
            D_opt=0
            l_p_zeros_l_psnr.append(p)
    else:
        for i in range(len(l_per)-2):
            if l_per[i]<l_per[i+1] or abs(l_per[i]-l_per[i+1])<0.01:
                D_opt=l_D[i]
                break
            elif l_per[i]==0:
                D_opt[i]=l_D[i]
                break
    l_opt.append(D_opt)
    d_patch_Dopt[p]=D_opt     # keep Dopt in a dictionary for each patch
    #print(D_opt)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")

# PLOT THE FFN % AGAINST D FOR A PATCH
# plt.plot(d_fnn.keys(),d_fnn.values())
# plt.plot(l_D[0:8],l_per,label="patch 155")
# plt.title("FNNper vs embedding D")
# plt.xlabel('D number')
# plt.ylabel('FNNper')
# plt.legend()

# PLOT THE HISTOGRAM OF COUNTS OF THE OPTIMAL D
from collections import Counter
l_opt = list(d_patch_Dopt.values())
counts = Counter(l_opt)                           # use counter to count the Dopt
d_counts = dict(counts)
d_counts_sorted = dict(sorted(d_counts.items()))  #sort the counts
D_numbers=list(d_counts_sorted.keys())            #make a list with the counts after sorting
x_labels=[]
for i in D_numbers:                               #define if Dman is high or low
    if i==25:
        i="Dmax H"
    if i==0:
        i="Dmax L"
    i=str(i)
    x_labels.append(i)
ap_times=list(d_counts_sorted.values())
plt.bar(D_numbers, ap_times,label="psnr_dif=5",color='darkblue') #bar plot for the histogram
plt.xlabel('Embedding dimension, D',fontsize=14)
plt.ylabel('Counts',fontsize=14)
plt.title('Histogram: Distribution of counts across D',fontsize=16)
plt.xticks(D_numbers,x_labels)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# PLOT AN IMAGE MAP OF Dopt WITH GRAYSCALE WITH GRID
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
image=frames[10]       # chose a frame to plot
height, width = image.shape
# Set up your figure and axis
fig, ax = plt.subplots()
grid_size=patch_size
ax.imshow(image, cmap='gray')
# Draw vertical lines at multiples of 16
for i in range(grid_size, width, grid_size):
    line = mlines.Line2D([i, i], [0, height], color='blue', linewidth=0.5)
    ax.add_line(line)
# Draw horizontal lines at multiples of 16
for i in range(grid_size, height, grid_size):
    line = mlines.Line2D([0, width], [i, i], color='blue', linewidth=0.5)
    ax.add_line(line)
# # Add numbers to the grid squares
# count = 0
# for i in range(0, height, grid_size):
#     for j in range(0, width, grid_size):
#         ax.text(j + 2, i + 6, str(count), color='red', fontsize=5)
#         count += 1
# #plt.show()  

gray_colors=len(d_counts_sorted)
start_range = 0
end_range = 255
full_range = np.arange(start_range, end_range + 1) #Create an array representing the range
split_ranges = np.array_split(full_range, gray_colors) #Split the range into equal parts
mean_gray_values = [round(np.mean(part)) for part in split_ranges] #keep only the mean values of the splitted ranges
i1=0
d_dopt_grayval={}
for i in d_counts_sorted:
    d_dopt_grayval[i]=mean_gray_values[i1]
    i1+=1

imag=frames[14] #chose a frame to plot
p_num=0
p_size=patch_size
for i in range(0,imag.shape[0]//p_size):        # loop through the pixels
    for j in range(0,imag.shape[1]//p_size):
        patch_dopt=d_patch_Dopt[p_num]          # find for the patch Dopt
        gray_value=d_dopt_grayval[patch_dopt]   # and the grey value for this Dopt
        print(patch_dopt)
        imag[(i*p_size):(i*p_size)+p_size,(j*p_size):(j*p_size)+p_size]=gray_value   #set all the pixels of the patch to the grey value   
        p_num+=1
plt.imshow(imag, cmap='gray')
#plt.colorbar(label='Grayscale Value')
cbar_ticks=np.linspace(0, 255, num=len(D_numbers))
cbar_labels= D_numbers
plt.colorbar(ticks=cbar_ticks, label='D Grayscale Values').ax.set_yticklabels(cbar_labels)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




"""(3) RQA WITH METRICS FOR EACH PATCH WITH THE DOPT....................................................................................................."""

#change all values of Dopt 0 to 1
for key, value in d_patch_Dopt.items():
    if value == 0:
        d_patch_Dopt[key] = 1

#try RQA for a patch
#psnr_rqa1(ar_pre_rqa[50],2,35,155,2,2)

from collections import Counter
import time
start_time = time.time()

# calculate the RQA metrics for each patch and store their valuew in the following lists
l_RR=[]
l_DET=[]
l_lavg=[]
l_ent=[]
l_TT=[]
l_lam=[]
l_Dopt=[]
d_patch_l_D_metrics={}
for p in range(ar_pre_rqa.shape[0]):
    print(p)
    dopt=d_patch_Dopt[p]
    l_Dopt.append(dopt)
    l_res=def_functions_rqaPSNR.psnr_rqa1(ar_pre_rqa[p],dopt,35,p,2,2)
    #if dopt==25:
    if l_res[0]>0.9:
        l_res.append("B")
    else:
        l_res.append("F")
    l_RR.append(l_res[0])
    l_DET.append(l_res[1])
    l_lavg.append(l_res[2])
    l_ent.append(l_res[3])
    l_TT.append(l_res[5])
    l_lam.append(l_res[4])

    l_res.insert(0,dopt)
    d_patch_l_D_metrics[p]=l_res

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")


#SAVE DICT as txt
# import json
# os. chdir("C:\\Users\\asd\Documents\\BrainSIM\\dict_list_tosave\\dict_dataset_forground_background")
# with open("d_ucf101_drumming.txt", 'w') as f:
#      f.write(json.dumps(d_patch_l_D_metrics))



##scatter plot D vs RR
plt.scatter(l_Dopt, l_RR)
plt.xlabel('D opt values')
plt.ylabel('RR values')
plt.title('D vs RR')

 
#plot histogram of metrics, chose metric
metric_values = l_RR        #chose metric
metric="RR"
# convert all items of list to integers
#for int
#metric_values = [int(x) for x in metric_values if isinstance(x, int) or x.isdigit()]
#for float
metric_values = [float(item) for item in metric_values]
# Define the number of bins for the histogram for each metric
num_bins = 10
maxval=max(metric_values)
if maxval<1:
    maxval=1
bin_edges = np.linspace(0, maxval, num=21)
# Plot the histogram
plt.hist(metric_values, bins=bin_edges,edgecolor='black')
plt.grid(True)
plt.xlabel('Value')
#plt.xlim(0,1)
#plt.xlim(0, max(metric_values) + 5) 
plt.ylabel('Frequency')
plt.title('Histogram of '+metric + " in patches")
plt.show()


### plot an image heatmap of metrics with grayscale with grid
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
#plot the grid
image=frames[10]
height, width = image.shape
# Set up your figure and axis
fig, ax = plt.subplots()
grid_size=patch_size
ax.imshow(image, cmap='gray')
# Draw vertical lines at multiples of 16
for i in range(grid_size, width, grid_size):
    line = mlines.Line2D([i, i], [0, height], color='blue', linewidth=0.5)
    ax.add_line(line)
# Draw horizontal lines at multiples of 16
for i in range(grid_size, height, grid_size):
    line = mlines.Line2D([0, width], [i, i], color='blue', linewidth=0.5)
    ax.add_line(line)
# Add numbers to the grid squares
# count = 0
# for i in range(0, height, grid_size):
#     for j in range(0, width, grid_size):
#         ax.text(j + 2, i + 6, str(count), color='red', fontsize=5)
#         count += 1

# put grayscale values of pathces
gray_colors=10
start_range = 0
end_range = 256
full_range = np.arange(start_range, end_range + 1)       # Create an array representing the range
split_ranges = np.array_split(full_range, gray_colors)   # Split the range into equal parts
mean_gray_values = [round(np.mean(part)) for part in split_ranges]
# Assign each value to the appropriate bin index
metric="RR"     #set the metric you want
lmetric=[float(item) for item in l_RR]
maxval=max(lmetric)
if maxval<1:
    maxval=1
minval=0
# Define the bin edges
bin_edges = np.linspace(minval, maxval, num=10)
bin_indices = np.digitize(lmetric, bins=bin_edges)-1

imag=frames[15]
p_num=0
p_size=patch_size
for i in range(0,imag.shape[0]//p_size):
    for j in range(0,imag.shape[1]//p_size):
        gray_value=mean_gray_values[bin_indices[p_num]]
        imag[(i*p_size):(i*p_size)+p_size,(j*p_size):(j*p_size)+p_size]=gray_value      
        p_num+=1
plt.imshow(imag, cmap='gray')
#plt.clim(0,256)
cbar_ticks=mean_gray_values#np.linspace(0, 255, num=10)
cbar_labels=list(np.round(np.linspace(0, maxval, 10),1))
plt.colorbar(ticks=cbar_ticks, label=metric+' in Grayscale Values',).ax.set_yticklabels(cbar_labels)
#cbar=plt.colorbar(label='Grayscale Value')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(metric + " values")
plt.show()




""" (4) REARANGE VIDEO FRAMES ACCORDING TO DOPT OR RR AND COMPRESS .........................................................................................................................."""


#create a dict number of patch : frames to keep the same for this patch 
#according to RR
d_patch_D_RR={}
nrr=0
for i in l_RR:
    if i<0.2:
        d_patch_D_RR[nrr]=1
    elif 0.2<=i<0.4:
        d_patch_D_RR[nrr]=2      
    elif 0.4<=i<0.6:
        d_patch_D_RR[nrr]=4      
    elif 0.6<=i<0.8:
        d_patch_D_RR[nrr]=8
    elif 0.8<=i<=1:
        d_patch_D_RR[nrr]=25
    nrr+=1


## create an array that will contain info about patch frame and flat pixel values
## but for every patch it will keep the same values for the Dopt or the RR of the patch
patch_frames_flatvalues=[]
n=0
for p in range(ar_pre_rqa.shape[0]): #run for each patch
    #Dopt_for_patch=d_patch_Dopt[p]  #for Dopt
    Dopt_for_patch=d_patch_D_RR[p]  #for RR    
    patch_fr=[]
    a=0
    print(n)
    for fr in range(ar_pre_rqa.shape[1]): #for each frame in each patch
        if Dopt_for_patch==1: ## if dopt=1 keep every frame
            patch_fr.append(ar_pre_rqa[p][fr])
            
        elif fr % Dopt_for_patch== 0: ## if dopt>1 every dopt frames keep the same frame
            for i in range(Dopt_for_patch):
                if fr + i > ar_pre_rqa.shape[1]-1: # make sure to not exceed the number of frames
                    break
                patch_fr.append(ar_pre_rqa[p][fr])
    n+=1
    patch_frames_flatvalues.append(patch_fr)

patch_frames_flatvalues=np.array(patch_frames_flatvalues)

##now rearange the patch_frames_flatvalues to reconstract the frames according to pathces

rec_frames=[]

p1=def_functions_rqaPSNR.patches_unique(patch_size,frames[1])[1] #rows of patches
p2=def_functions_rqaPSNR.patches_unique(patch_size,frames[1])[2] #column of patches



f_p_v_comp=patch_frames_flatvalues.transpose(1, 0, 2)   #rearange patch flat values (frames,patches,pixel values)
p_shape=(patch_size,patch_size)
for f in range(f_p_v_comp.shape[0]):
    print(f)
    reconstructed_frame = []
    reconstructed_frame = np.zeros((p1*patch_size,p2*patch_size))  #create empty reconstrated patch
    reconstructed_frame = np.array(reconstructed_frame)
    for pp in range(f_p_v_comp.shape[1]):
        p_sq=f_p_v_comp[f][pp][:patch_size*patch_size].reshape(p_shape)


        row = pp // p2
        col = pp % p2

        # Calculate the coordinates in the frame
        start_row = row * patch_size
        end_row = start_row + patch_size
        start_col = col * patch_size
        end_col = start_col + patch_size

        # Place the patch in the frame
        reconstructed_frame[start_row:end_row, start_col:end_col] = p_sq
    rec_frames.append(reconstructed_frame)

rec_frames=np.array(rec_frames)

rec_frames[0]


######compress the reconstracted frames.....................................................
import cv2
os.chdir("C:\\Users\\asd\Documents\\BrainSIM\\video\\UCF101_videos\\UCF101")

input_video=vid_name
# Set output video file name
#output_video = "g" + vid_name[1:-4]+"_comp_fnnDopt_psize"+str(patch_size) +".mp4"
output_video = "g" + vid_name[1:-4]+"_comp_RR_psize"+str(patch_size) +".mp4"


# # Open input video file
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
# width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_video, fourcc, fps, (rec_frames.shape[2], rec_frames.shape[1]))

# Write each frame to the video file
a=0
for frame in rec_frames:
    # Convert to 8-bit depth
    # plt.imshow(frame, cmap='gray')
    # plt.show()
    a+=1
    frame_8bit = cv2.convertScaleAbs(frame)

    # Write the converted frame to the video file
    out.write(frame_8bit)
# Release the VideoWriter object
out.release()
 

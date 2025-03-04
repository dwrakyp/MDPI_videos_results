**INTRODUCTION**

RQA in video can be used for two different applications. One is RQA in full frames for scene change detection,
and the other one is RQA for motion detection in each patch of the grid of patches we create for the frames.
The scripts for those two applications can be found here.

**SCRIPTS DESCRIPTION**

There are two scripts where all the functions are difined one script for RQA in full frames (def_functions_rqafullframes.py)
and another one for RQA in patches (def_functions_rqaPSNR.py).\
**rqafullframes.py**           --> identify frame change with RP from RQA and store the frames of change in a list\
**compare_RQA_with_GT**        --> compare the results of scene change from RQA with the ground throuth of scene change for the video\
**RQApsnr_fnnDopt_patches.py** --> run RQA for each patch after defining the Dopt with fnn, show the results with plots

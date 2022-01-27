# LVMM model
the intermediate layder guide（LVMM） is inspired by the article [Learning-based Video Motion Magnification]
We use (an unofficial implementation of [Learning-based Video Motion Magnification]in Pytorch==1.3.0.) 
code and parameter weights, which is a very nice work and helped me a lot.

#Code
Our code is stored in CODE.(in Pytorch==1.6.0).

M-MEMN(Multi-layer micro-expression magnification network)
*datasets* is the processing of dataset.*networks* and  *magnet*  contains our important network mode.
*train_and_eval* is our main function.*evaluate* is our test function.*callbacks* represents saving and loading parameters.
*S-MEMN* is the pretraining  weight of teacher-model(LVMM),S-MEMN and M-MEMN

LPS-MSA(local pixel-relation sliding based multi-head selfattention)
*models*  contains our important network mode.*train*is our main function.
If you need to run the program, please modify the data input by yourself

# Results
The magnified results of the three datasets are stored in *result* . 'amp_factor=A' is the onset frame, 'amp_factor=B' is the apex frame, 
'amp_factor=10' is  the magnification factor  α=10 of each S-MEMN in the M-MEMN.
In order to better observe the magnification of motion. We present our experimental results in the form of PPT(original_magnification.ppt). 
(1)magnification denotes the magnification factor  α=10 of each S-MEMN in the M-MEMN.
(2)Apex denotes the apex in the three datasets.

#Limitations
(1)In the three datasets，some images of onset frame and  apex frame motion amplitude is small, 
causes the output magnified images change is not obvious.But most of the other magnification picture is obvious.
(2)Due to the small training set, the picture has a limitation of clarity. But the magnification change is obvious.

#Notice
Due to size constraints, we only provide part of the training data set.
All databases in this article are from public databases and were agreed to use them.


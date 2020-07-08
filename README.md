# Unlearning Scanner Bias for MRI Harmonisation
## Code for implementation of Unlearning Scanner Bias for MRI Harmonisation

If you use code from this repository please cite the appropriate paper: 

Age Prediction Task (MICCAI 2020): Add Link

Segmentation Task (MIUA 2020): Add Link

Software Versions
-----------------
Python 3.5.2

PyTorch 1.0.1.post2

Age Prediction Task (MICCAI 2020)
---------------------------------
![GitHub Logo](/figures/network_architecture.png)

Code supplied was used for the age prediction task but the framework is general and should be applicable to any feedforwards network. The architecture used needs to have three sections as shown in the figure above:
  - Feature extractor
  - Label predictor
  - Domain predictor

Scripts
-------
- Fully supervised Training (Normal Supervised) 
  - Three datasets with training labels available for all datasets. 
- Biased Distributions (Normal Biased)
  - Two datasets with different distributions for the main task label. Unlearning is completed on the overlap subjects.


GM/WM Segmentation Task (MIUA 2020)
---------------------------------
![GitHub Logo](/figures/seg_network_new.png)

Code used for the segmentation task for GM/WM/CSF segementation. Architecture is based on the UNet architecture. 

Scripts
-------
- Fully supervised training (Segmentation Unlearning)
  - Two datasets with training labels available for all datasets. 
- Semisupervised Training (Segmentation Semisupervised)
  - Two datasets, limited labels available for one dataset. All data points used for unlearning.
- Multiple unlearning points (Segmentation Unlearning Multi)
  - Domain predictor located at both final convolution and the bottleneck to explore effect of unlearning location. 

  











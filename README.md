# Unlearning Scanner Bias for MRI Harmonisation
## Code for implementation of Unlearning Scanner Bias for MRI Harmonisation

### Journal Paper Published: https://doi.org/10.1016/j.neuroimage.2020.117689

This is a working release of the code for unlearning scanner bias for MRI Harmonisation. Any issues please contact: nicola.dinsdale@dtc.ox.ac.uk. Further code will be added in time. 

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


If you use code from this repository please cite the appropriate paper: 

Age Prediction Task (MICCAI 2020): https://link.springer.com/chapter/10.1007/978-3-030-59713-9_36

Segmentation Task (MIUA 2020): https://link.springer.com/chapter/10.1007/978-3-030-52791-4_2

Conference Presentation: https://www.youtube.com/watch?v=CI59VLCwDVA&feature=youtu.be

Citation:

```@article{DINSDALE2021117689,
title = {Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal},
journal = {NeuroImage},
volume = {228},
pages = {117689},
year = {2021},
doi = {https://doi.org/10.1016/j.neuroimage.2020.117689},
author = {Nicola K. Dinsdale and Mark Jenkinson and Ana I.L. Namburete}
}

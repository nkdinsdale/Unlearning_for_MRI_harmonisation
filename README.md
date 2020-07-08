# Unlearning Scanner Bias for MRI Harmonisation
## Code for implementation of Unlearning Scanner Bias for MRI Harmonisation

Age Prediction Task (MICCAI 2020): Add Link

Segmentation Task (MIUA 2020): Add Link

Age Prediction Task (MICCAI 2020)
---------------------------------
![GitHub Logo](/figures/network_architecture.png)

Code supplied was used for the age prediction task but the framework is general and should be applicable to any feedforwards network. The architecture used needs to have three sections as shown in the figure above:
  - Feature extractor
  - Label predictor
  - Domain predictor

- Fully supervised Training (Normal Supervised) 
  - Three datasets with training labels available for all tasks. 
- Biased Distributions
  - Two datasets with different distributions for the main task label. Unlearning is completed on the overlap subjects. 
  











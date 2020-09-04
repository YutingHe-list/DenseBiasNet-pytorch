# DenseBiasNet-pytorch
The dense biased network implementated by pytorch has two versions:

1. DenseBiasNet_a.py, DenseBiasNet_a.py, DenseBiasNet_state_a_deeper and DenseBiasNet_a_deeper.py are implementated via the dense biased connection in our MICCAI version.

2. DenseBiasNet_state_b.py, DenseBiasNet_state_b_deeper.py and DenseBiasNet_b.py are implementated via the dense biased connection in our MedIA version.

When 'state' in the file name, the bias quantity $m$ in the dense bias connection is the same fixed value such as 1, 2, etc in each layer. When there is no 'state' in file name, the bias quantity $m$ in the dense bias connection is the proportion of the feature maps from each layer.

If you use densebiasnet or some part of the code, please cite (see bibtex):
* MICCAI version:
**DPA-DenseBiasNet: Semi-supervised 3D Fine Renal Artery Segmentation with Dense Biased Network and Deep Priori Anatomy**  [https://doi.org/10.1007/978-3-030-32226-7_16](https://www.researchgate.net/publication/335866931_DPA-DenseBiasNet_Semi-supervised_3D_Fine_Renal_Artery_Segmentation_with_Dense_Biased_Network_and_Deep_Priori_Anatomy) 

* MedIA version:
**Dense biased networks with deep priori anatomy and hard region adaptation: Semi-supervised learning for fine renal artery segmentation**
[https://doi.org/10.1016/j.media.2020.101722](https://www.sciencedirect.com/science/article/pii/S1361841520300864)

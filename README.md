# StratGAN

Coupling a generative adversarial network with image quilting to produce basin-scale realizations of fluvial stratigraphy.

<img src="https://github.com/amoodie/stratgan/blob/master/private/basin_demo.gif" alt="basin_demo_gif">


This was a project that began out of an interest to educate myself on machine learning topics, in association with a [class I organized](http://andrewjmoodie.com/2018/12/machine-learning-seminar/).
I have decided to open source the project now, along with a brief description, so that it can be viewed as part of my professional portfolio.
I hope to detail the model in a manuscript in the future. 

## Abstract
Subsurface reservoir size estimates involve considerable uncertainty, which impacts the quality of  reserve size and valuation models.
Conventional rules-based and process-based numerical models are useful to model this uncertainty, but involve simplifying assumptions about depositional environments and reservoir geology that may be poorly constrained.
Generative adversarial neural networks (GANs) are a machine learning model that is optimized to produce synthetic data that are indistinguishable from an arbitrary training dataset, and are an attractive tool for modeling subsurface reservoirs.
We have developed a generative adversarial network that is trained on laboratory experimental stratigraphy and produces realizations of basin-scale reservoir geology, while honoring ground-truth well log data. 
In this way, StratGAN reduces subsurface uncertainty through a large sampling of realistic potential rock geometries throughout a reservoir, without any a priori assumptions about the geology.  
StratGAN couples a deep-convolutional generative adversarial network (DCGAN) with an image quilting algorithm to scale up channel-scale realizations of reservoir geometries to basin-scale realizations.


## Model description
The model is a deep-convolutional generative adversarial network (DCGAN) that has been trained on laboratory experiment data. 
I use custom `tensorflow` implementations of convolutional layers and dense layers, which include my flexible batch normalization operation.
I have implemented Efros-Freeman minimum-cost-boundary patch quilting, using patches from the GAN (channel scale). 
The patches fed to the E-F algorithm are selected based on conditional inpainting methods (e.g., Dupont et al., 2018).
I have produced routines to include ground-truth data in the basin scale realizations, such as vertical core-logs that record channel and non-channel intervals.

## Dependencies
 * Python3
 * Tensorflow <2
 * scipy, numpy
 * matplotlib


## References
* Dupont et al., 2018; [https://arxiv.org/abs/1802.03065](https://arxiv.org/abs/1802.03065)

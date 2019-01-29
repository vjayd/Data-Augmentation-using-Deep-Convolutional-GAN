# Face Generation using Deep Convolutional Generative Adversarial Network 

This repository contains code to generate faces from the given [celebrity dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

# Steps to run the code
1. Git clone the folder.
2. Download and extract the celeb dataset into data folder.
3. Run python train.py from terminal.


You can also use this code for data augmentation just replace the celeb dataset with your dataset folder. Although GAN's have been found to be unstable while training re running the same code with same or different parameters will give you different resutls.

## References: 
[GAN](https://arxiv.org/abs/1406.2661)
[Data Augmentation] (https://arxiv.org/pdf/1609.08764.pdf)
[Pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

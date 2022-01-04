# Introduction

This repository is the continuation of my winter research project, where I collected AUSLAN data and labeled it, so I can use it as part of a sign language detector. The code in this repository has been forked from the paper "Sign Language Detection 'In the Wild' with Recurrent Neural Networks". It has been changed to work with Tensorflow 2.5.0. And a few other changes to allow it to work with my dataset.

# Project Details
The goal of this project is to use Neural Networks for AUSLAN Detection and to identify the person signing. The dataset being used is news conferences from ABC and 7News with AUSLAN iterpreters. Technical details about the model's architecture can be found in [1].

In order to identify the person signing I'm using the Mask-RCNN trained on the Microsoft Coco dataset to detect humans within the video, and passing it to the AUSLAN detector. The code for the demo can be found in the "demo" foder along with instructions on how to run it.

An example video output from the demo can be found [here](https://youtu.be/E0k3QGOsR6Q)

For a literature review, and a rundown on the dataset read the pdf in the folder "report"


# References

[1] Borg, M., & Camilleri, K. P. (2019). SIGN LANGUAGEDETECTION “IN THE WILD” WITHRECURRENT NEURAL NETWORKS.https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6836575&tag=1

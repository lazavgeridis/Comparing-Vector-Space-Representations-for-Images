
# Introduction

This is the 3rd and final project of Software Development for Algorithmic Problems. Part 1 and 2 can be found here and here respectively.  
In this project, we compare 2 different vector space representations for the MNIST dataset, and evaluate how they perform on nearest neighbor
search, clustering, and classification. 


# Description

We start off by loading a pretrained autoencoder model (Part 2), and basically converting each image (28x28 pixels) to a 10d latent space 
representation, as show below:

![Screenshot](images/ae_reduce.png)


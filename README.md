
# Introduction

This is the 3rd and final project of Software Development for Algorithmic Problems. In this project, we compare 2 different vector space 
representations for the MNIST dataset, and evaluate how they perform on nearest neighbor search and clustering. 


# Description

### Reducing Data Dimensionality

We used a (mirrored) CNN autoencoder architecture which includes a bottleneck layer in the middle. After training the model, we "feed" each image
to the network as input, we take the output of this bottleneck layer and use it as a new vector representation for this image. In our case, this 
bottleneck layer produces 10 output values so the above procedure can be expressed more formally as a mapping: 784d space --> 10d space:

![Screenshot](images/ae_reduce.png)

### NN Search 

## Approximate NN (784d) vs Exact NN (784d) vs Exact NN (10d) 


## Exact NN (784d) : Manhattan vs Earth Mover's Distance



### Clustering in the 2 Vector Spaces




# Introduction

This is the 3rd and final project of Software Development for Algorithmic Problems. In this project, we compare 2 different vector space 
representations for the MNIST image dataset, and evaluate how they perform on nearest neighbor search and clustering.


# Description

## Part 1 : Reducing Data Dimensionality

We used a (mirrored) CNN autoencoder architecture which includes a bottleneck layer in the middle. After training the model, we "feed" each image
to the network as input, we take the output of this bottleneck layer and use it as a new vector representation for this image. In our case, this 
bottleneck layer produces 10 output values so the above procedure can be expressed more formally as a mapping:  **784d space -> 10d space** :

![Screenshot](images/ae_reduce.png)

With the completion of part 1 the new (latent space) representations
for the MNIST training and test sets will be stored in the 2 output files
specified by the user in the command line. These 2 files are located in 
[output_files/output_dataset_file](output_files/output_dataset_file) and 
[output_files/output_query_file](output_files/output_query_file) respectively.

(**update**: producing 20 output values instead of 10 in the bottleneck layer seems to be working better)

## Part 2 : NN Search 

### 2A : Approximate NN (784d) vs Exact NN (784d) vs Exact NN (10d) 
We have already conducted a similar experiment in Project 1 and we knew that using LSH for Approximate NN Search was already performing
very well (~1 approximation ratio and a lot faster than Exact NN search) . This time, we added Exact NN Search in the reduced vector space to 
our comparisons. Its approximation factor is higher as expected (~1.9), but the search time is almost halved compared to LSH. 


### 2B : Exact k-NN (784d) : Manhattan vs Earth Mover's Distance

In this section, we implemented earth mover's distance and compared it to manhattan distance, our default metric for measuring similarity 
between 2 images up to this point. For each image in the query set, its 10 nns are found using both metrics. The files containing labels 
are used here in order to measure the accuracy of each approach. It turned out that for MNIST, manhattan distance is both more accurate 
and substantially faster.  
To compute the earth mover's distance between 2 images, we had to minimize an objective function with respect to some constraints. For this
purpose, [google or-tools](https://developers.google.com/optimization) was used. To download and install OR-Tools for C++, refer to the official installation 
[page](https://developers.google.com/optimization/install/cpp).


## Part 3 : Clustering in the 2 Vector Spaces

In this section, we compared 3 different clustering procedures:  
1. clustering in 10d vector space + computing silhouette and objective function in 784d vector space
2. clustering in 784d vector space + computing silhouette and objective function in 784d vector space
3. create clusters using cnn's class predictions for the training set (784d) + computing silhouette and objective function in 784d vector space

The evaluation of each procedure (silhouette and objective) had to be made in the original vector space for our comparisons to make sense. We 
observed that number (1) produced the worst results, which is something we expected because of the reduced dimensionality of the data. Procedures
(2) and (3) performed similarly but (2) achieved slightly higher silhouette score and lower clustering objective value.


# Execution
For **part 1**, the program to execute is `reduce.py`. After navigating to
[src/python](src/python) you can execute the program as:  
```
$ python3 reduce.py --dataset ../../datasets/train-images-idx3-ubyte 
                    --queryset ../../datasets/t10k-images-idx3-ubyte
                    -od <output_dataset_file>
                    -oq <output_query_file>
```  
For **part 2A**, you need to first navigate to the [src/cpp/search](src/cpp/search) directory and
then execute the following commands:  
```
$ make  
$ ./search -d ../../../datasets/train-images-idx3-ubyte 
           -i ../../../output_files/output_dataset_file
           -q ../../../datasets/t10k-images-idx3-ubyte
           -s ../../../output_files/output_query_file
           -k 4
           -L 5
           -o output
```
To delete the generated object files, output file and the executable file
simply run  
```
$ make clean
```  
To run **part 2B**, first make sure you have installed Google OR-Tools for C++ (see
above). Then, navigate to the directory where you installed or-tools (under this directory
you should also see the `Makefile` as well as the directories `examples`, `include`, `lib` `objs`, etc) and run:
```
$ make DEBUG='-Ofast' build
SOURCE=relative/path/to/Comparing_Vector_Space_Representations_for_Images/src/cpp/emd/search.cc
``` 
The generated executable file is located by default in the `bin` directory, so
to execute, run the following:
```
$ cd ./bin  
$ ./search -d relative/path/to/Comparing_Vector_Space_Representations_for_Images/datasets/train-images-idx3-ubyte
           -q relative/path/to/Comparing_Vector_Space_Representations_for_Images/datasets/t10k-images-idx3-ubyte
           -l1 relative/path/to/Comparing_Vector_Space_Representations_for_Images/datasets/train-labels-idx1-ubyte
           -l2 relative/path/to/Comparing_Vector_Space_Representations_for_Images/datasets/t10k-labels-idx1-ubyte
           -o relative/path/to/Comparing_Vector_Space_Representations_for_Images/output_files/{output_file_name}
           -EMD
```
Lastly, to execute **part 3**, navigate to the [src/cpp/cluster](src/cpp/cluster) directory and run:
```
$ make
$ ./cluster -d ../../../datasets/train-images-idx3-ubyte
            -i ../../../output_files/output_dataset_file
            -n ../../../output_files/nn_clusters
            -c ../../../include/cluster/cluster.conf
            -o ../../../output_files/{output_file_name}
```

# Quantum Autoencoder
This is an ongoing examination project on Quantum Machine Learning. The goal is to solve the classification problem of the MNIST database of handwritten digits using a Quantum Autoencoder.

# Table of contents
1. [The MNIST database](#The-MNIST-database)
   1. [Isomap](#Isomap)
2. [Classical ML Classifiers](#Classical-ML-Classifiers)
   1. [Gaussian Classifier](#Gaussian-Classifier)
   2. [Random Forest Classifier](#Random-Forest-Classifier)

Projected steps to complete the project:
1) Import the MNIST databse and take a look at it
2) Use some classical ML algorithm to solve the classification problem 
3) Reduce the dimensionality of the photos using PCA, maxpooling or average pooling
4) Figure it out how to encode and decode blocks of the images separately using a Quantum Autoencoder

## The MNIST database
The MNIST database is a database of 70000 handwritten digit. We can import it and divide in two sets. The first set consisting of 60000 images will be used as the train set and the remaining 10000 images will be used as the test set. 

Let's take a look at some of the images contained in the database:

![title](Images/MNIST.png)

### Isomap
Each image in the database is 28 x 28 = 784 pixels. We treat each pixel in the images as a feature, thus we have 784 features. It is difficult to visualize our data points in a 784-dimensional parameter space so we make use of the manifold learning algorithm _Isomap_ to reduce the dimensionality to 2 and gain more insight on the structure of the database.

Here is the plot of 1/20 of the database in the 2-dimensional parameter space

![title](Images/MNIST_Isomap.png)

We can plot singularly every digit to understand the variety of forms that the digits can take, here is the plot of the 1's in the 2D parameter space.

![title](Images/MNIST_Isomap_1.png)

## Classical ML Classifiers

### Gaussian Classifier 

![title](Images/Confusion_Matrix_Gaussian.png)

### Random Forest Classifier 

![title](Images/Confusion_Matrix_RandomForest.png)

## PCA reduction 
![title](Images/PCA_evr.png)

<p align="middle">
  <img src="Images/pca_128_reconstructed.png"  width="32%" />
  <img src="Images/pca_64_reconstructed.png" width="32%" /> 
  <img src="Images/pca_128_reconstructed.png" width="32%" />
</p>

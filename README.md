# MNIST handwritten digits classification problem using a Quantum Autoencoder
This is an ongoing examination project on Quantum Machine Learning. The goal is to solve the classification problem of the MNIST database of handwritten digits using a Quantum Autoencoder.

# Table of contents
1. [The MNIST database](#The-MNIST-database)
   1. [Isomap](#Isomap)
2. [Classical ML Classifiers](#Classical-ML-Classifiers)
   1. [Gaussian Classifier](#Gaussian-Classifier)
   2. [Random Forest Classifier](#Random-Forest-Classifier)
3. [PCA Reduction](#PCA-reduction)
4. [Quantum Autoencoder](#quantum-autoencoder)
   1. [Ansatz](#ansatz)
   2. [Feature map](#feature-map)
   3. [Swap Test](#swap-test)
   4. [Results](#results)

## The MNIST database
The MNIST database is a database of 70000 handwritten digits. We can import it and divide in two sets. The first set consisting of 60000 images will be used as the train set and the remaining 10000 images will be used as the test set. 

Let's take a look at some of the images contained in the database:

<p align="center">
<img src="Images/MNIST.png" alt="drawing" width="60%"/>
</p>

### Isomap
Each image in the database is 28 x 28 = 784 pixels. We treat each pixel in the images as a feature, thus we have 784 features. It is difficult to visualize our data points in a 784-dimensional parameter space so we make use of the manifold learning algorithm _Isomap_ to reduce the dimensionality to 2 and gain more insight on the structure of the database.

Here is the plot of 1/20 of the database in the 2-dimensional parameter space

<p align="center">
<img src="Images/MNIST_Isomap.png" alt="drawing" width="60%"/>
</p>

We can plot singularly every digit to understand the variety of forms that the digits can take, here is the plot of the 1's in the 2D parameter space.

<p align="center">
<img src="Images/MNIST_Isomap_1.png" alt="drawing" width=60%/>
</p>

This result gives us an insight of the variety of different forms that the number "1" can take inside the database, the same applies to the othe digits.

## Classical ML Classifiers
We try some Classical ML Classifiers to compare then the results with our Quantum Classifier.

### Gaussian Naive Bayes Classifier 

As a first try we use a Gaussian Naive Bayes Classifier, which works under the assumpion that the data from each classes are derived from a Gaussian distribution. Of course this could not be the case, in fact we obtain very poor result:

 Accuracy score = 0.5558 

We can plot the confusion matrix to better understand the errors made by this classifier.

<p align="center">
<img src="Images/Confusion_Matrix_Gaussian.png" alt="drawing" width="60%"/>
</p>

As we can see many digits are misclassified by this simple method.

### Random Forest Classifier 
A more sophisticated method is to use a Random Forest Classifier, an ensamble learner built on decision trees. This time we obtain good result:


<center >

 | Class | Precision | Recall | f1-score | Support |
 |:---:  | :---      | :---   | :---:    | :---:   | 
 | 0     |     0.99  |   0.97 |    0.98  |   997   | 
 | 1     |     0.99  |   0.99 |    0.99  |  1134   | 
 | 2     |     0.97  |   0.96 |    0.97  |  1039   | 
 | 3     |     0.97  |   0.96 |    0.97  |  1014   | 
 | 4     |     0.97  |   0.98 |    0.98  |   979   |
 | 5     |     0.97  |   0.98 |    0.97  |   880   |
 | 6     |     0.98  |   0.98 |    0.98  |   962   |
 | 7     |     0.97  |   0.97 |    0.97  |  1023   |
 | 8     |     0.96  |   0.96 |    0.96  |   967   |
 | 9     |     0.95  |   0.96 |    0.96  |  1005   |

</center>

As before we plot the confusion matrix, this time we can see that the grat majority of digits are correctly classified.

<p align="center">
<img src="Images/Confusion_Matrix_RandomForest.png" alt="drawing" width="60%"/>
</p>

## PCA reduction 
To reduce the dimensionality of the dataset we make use of the Principal Component Analysis to quantify the relationship among the data and find a list of the principal axes in the data. Fitting with this method returns the _components_ and the _explained variance_. The _components_ can be seen as the directions of the vectors (principal axes) in the parameter space and the _explained variance_ as the squared-length of these vectors (measures how important an axes is in describing the distribution of the data). Thus PCA allow us to zeroing out the smallest principal components and reduce the dimensionality of our dataset.
If we think of every images as an array of 784 pixels we can write a "basis" for this space in the form of

$image(x) = x_1 \cdot (pixel \space 1) + x_2 \cdot (pixel \space 2) + \cdots + x_{784} \cdot (pixel \space 784)$.

We can think of a different basis like

$image(x) = mean  + x_1 \cdot (basis \space 1) + \cdots + x_{784} \cdot (basis \space 784)$.

PCA can be thought as a process that allows us to choose  optimal basis functions, such that some of them alone are enought to reconstruct the original image.

To estimate how many components do we need to suitably describe the data we plot the comulative _explained variance ratio_ as a function of the number of components.

<p align="center">
<img src="Images/PCA_evr.png" alt="drawing" width="60%"/>
</p>

As we can see from the images about 100 components are needed to retain 90% of the variance, thus loosing about 10% of the original information.
To see this we compare three images: the image on the left is the original digit in the database, the central images is the digit reduced with PCA to 64 features and then reconstructed, the images on the right is the same digit reduced with PCA to 128 features and then reconstructed.

<p align="middle">
  <img src="Images/pca_original.png"  width="32%" />
  <img src="Images/pca_64_reconstructed.png" width="32%" /> 
  <img src="Images/pca_128_reconstructed.png" width="32%" />
</p>

It is clear that the image is quite well reconstructed and we do not loose too much information reducing the database from 784 features to 128 or 64 features.

## Quantum Autoencoder

### Ansatz

### Feature map

### Swap Test

### Results

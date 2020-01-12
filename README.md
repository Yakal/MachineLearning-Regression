# MachineLearning-Regression
MachineLearning-Homework#2
Furkan Yakal 59943
Homework # 2 Discrimination by Regression
In this homework we tried to used discrimination by regression algorithm for multiclass classification. There are 1000 images from 5 different classes such as t-shirt, trouser, dress, sneaker and bag. In this homework our main purpose is to optimize the weight parameters for performing better predictions.
Using the given initial set of weight parameters, we performed our labeling predictions for our training set using sigmoid function. Sigmoid function basically maps any given input between [0, 1].
We matrix multiplied our training set with weights W then added bias vector to each column of our matrix. In our training set we have 500 images with 784 pixels. W matrix is in form of (784,5) each row represents the pixel value of 5 different classes. w0 is in the form of (5,1) where there is a bias term for each class. After having our first prediction we calculated the error function. But before doing that label vector should be reformed in to the matrix representation. For example, if the image is labeled as ‘2’ then the second column of the label matrix should be one and for other classes it should be zero. (N is number of images in the train set, C is the class number)
=:
𝐸𝑟𝑟𝑜𝑟 = 0.5 𝑥 (+ +(𝑦-,/ − 𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛-,/ )9 -;< /;<
By taking derivative of the error function with respect to W and w0 using chain rule, we found the gradients with the purpose of optimizing the W and w0.
𝜕𝐸𝑟𝑟𝑜𝑟 A
𝜕𝑊 = 𝑋 (𝑦 − 𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛) ∗ (𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛) ∗ (1 − 𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛)
=
= +(𝑦−𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛)∗(𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛)∗(1−𝑦𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛) -;<
  𝜕𝐸𝑟𝑟𝑜𝑟 𝜕𝑤0
 
Gradient basically shows us to direction in which we can decrease the error. With the stepsize hyperparameter which we determined at the beginning, we updated the W and w0, and we kept doing them for maximum number of iterations. (We set maximum number of iterations as 500 at the beginning.)
𝑊 = 𝑊𝑜𝑙𝑑 + 𝜁 𝜕𝐸𝑟𝑟𝑜𝑟 𝜕𝑊
𝑤0 = 𝑤0𝑜𝑙𝑑 + 𝜁 𝜕𝐸𝑟𝑟𝑜𝑟 𝜕𝑤0
After optimizing our weight for max-number-of-iteration times, our program gets ready for performing its predictions.
In the last step of our homework, we predict the labels for our train and test matrices using the sigmoid function with the updated weights. Managed to get better results.
  

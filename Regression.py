__author__ = "Furkan Yakal"
__email__ = "fyakal16@ku.edu.tr"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Read the images and labels
images = pd.read_csv("hw02_images.csv", header=None)
labels = pd.read_csv("hw02_labels.csv", header=None)

# training set is formed from first 500 images
train_set_images = np.array(images[:500])  # (500, 784)
train_set_labels = np.array(labels[:500])  # (500, 1)

# test set is formed from remaining 500 images
test_set_images = np.array(images[500:])  # (500, 784)
test_set_labels = np.array(labels[500:])  # (500, 1)

# extracting known parameters
num_train_data = train_set_labels.shape[0]
num_test_data = test_set_labels.shape[0]
number_of_classes = len(np.unique(train_set_labels))

# hyperparameters
eta = 1e-4
epsilon = 1e-3
max_iteration = 500

# initialize weight parameters
W = np.array(pd.read_csv("initial_W.csv", header=None))  # (784, 5)
w0 = np.array(pd.read_csv("initial_w0.csv", header=None)).reshape(number_of_classes)  # (5, )


# reformats the train labels in to 0-1 matrix format
def reformat_label_matrix():
    y_correct = np.zeros((num_train_data, number_of_classes))
    for i in range(num_train_data):
        y_correct[i, train_set_labels[i] - 1] = 1
    return y_correct


# sigmoid function, using in making predictions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# using the error, updates the W and w0 matrices, plots the error graph
def error_and_weight_updates(X, W, w0):
    y_correct = reformat_label_matrix()

    error_history = []
    for i in range(max_iteration):
        y_prediction = sigmoid(X.dot(W) + w0)

        error = y_correct - y_prediction
        squared_error = 0.5 * np.sum(error ** 2)
        error_history.append(squared_error)

        dv = error * y_prediction * (1 - y_prediction)
        gradient_W = np.matmul(X.T, dv)
        gradient_w0 = np.sum(dv, axis=0)

        W += eta * gradient_W
        w0 += eta * gradient_w0

    plot.plot(error_history)
    plot.xlabel('Iteration')
    plot.ylabel('Error')
    plot.show()


# predicts the class of each image using the updated W and w0
def predict_class(image_set, W=W, w0=w0):
    return np.argmax(sigmoid(image_set.dot(W) + w0), axis=1)


# generates the confusion matrix
def create_confusion_matrix(image_set, label_set):
    confusion_matrix = np.zeros((number_of_classes, number_of_classes))
    predictions = predict_class(image_set)
    for i in range(len(label_set)):
        confusion_matrix[predictions[i], label_set[i] - 1] += 1
    return confusion_matrix


# prints the matrix in desired format
def print_confusion_matrix(data_set_type, matrix):
    print("\n----------------------{}----------------------\n".format(data_set_type))
    labeled_conf_matrix = pd.DataFrame(matrix.astype(int),
                                       index=['T-shirt', 'Trouser', 'Dress', 'Sneaker', 'Bag'],
                                       columns=['T-shirt', 'Trouser', 'Dress', 'Sneaker', 'Bag'])
    print(labeled_conf_matrix)


# shape of the all data helpful while performing matrix operations
def shapes():
    print("Train_set_images: {}".format(train_set_images.shape))
    print("Train_set_labels: {}".format(train_set_labels.shape))
    print("Test_set_images: {}".format(test_set_images.shape))
    print("Test_set_images: {}".format(test_set_labels.shape))
    print("W: {}".format(W.shape))
    print("w0: {}".format(w0.shape))


if __name__ == "__main__":
    # shapes()
    error_and_weight_updates(train_set_images, W, w0)

    train_matrix = create_confusion_matrix(train_set_images, train_set_labels)
    print_confusion_matrix("TRAIN", train_matrix)

    test_matrix = create_confusion_matrix(test_set_images, test_set_labels)
    print_confusion_matrix("TEST", test_matrix)

import struct
from struct import unpack
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
      print("start of parsing X")
      byte = f.read()

      magicNumberHeader=unpack('>iiii',byte[0:16])
      assert magicNumberHeader[0]==2051
      byte=byte[16:]
      numofPics=magicNumberHeader[1];
      picSize=magicNumberHeader[2]*magicNumberHeader[3]
      
      X=np.zeros((numofPics, picSize), dtype=np.float32)
      for i in range(numofPics):
        pic=unpack(str(picSize)+'B', byte[i*picSize:(i+1)*picSize])
        pic=np.array(pic)
        X[i]=pic
      # print(np.linalg.norm(X[:10]))
      X = X/255
    
      # print(X[0])
      print("end of parsing X")

    with gzip.open(label_filename, 'rb') as f:
      print("start of parsing y")
      byte = f.read()

      magicNumberHeader=unpack('>ii',byte[0:8])
      assert magicNumberHeader[0] == 2049
      byte=byte[8:]
      numOfLabels=magicNumberHeader[1];
      
      labs=unpack(str(numOfLabels)+'B', byte)
      y=np.array(labs,dtype=np.uint8) 
      
      print("end of parsing y")
    return X,y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - np.choose(y, Z.T))
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n_iterations = int(X.shape[0] / batch)
    for i in range(n_iterations):
      X_batch = X[i * batch:(i + 1) * batch] # (batch * input_dim)
      y_batch = y[i * batch:(i + 1) * batch] # (batch * input_dim)
      # print("Helllo")
      exp_X_theta = np.matmul(X_batch, theta)
      exp_X_theta = np.exp(exp_X_theta) # (batch x num_classes)
      
      Z = exp_X_theta / np.sum(exp_X_theta, axis=1)[:,None] # (batch x num_classes)
      I_y = np.zeros_like(Z)
      np.put_along_axis(I_y, y_batch[:,None], 1, axis=1)  # (batch x num_classes)
      
      grad_softmax = np.matmul(X_batch.T, (Z - I_y)) / batch
      theta -= lr * grad_softmax
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # print(y)
    W1_0 = W1.copy()
    n_iterations = int(X.shape[0] / batch)
    # grad_W1 = np.zeros_like(W1)
    # grad_W2 = np.zeros_like(W2)
    # Z1 = np.zeros((batch, W1.shape[1]))
    for i in range(n_iterations):
      X_batch = X[i * batch:(i + 1) * batch] # (batch * input_dim)
      y_batch = y[i * batch:(i + 1) * batch] # (batch * input_dim)

      Z1 = np.matmul(X_batch, W1) # (batch * hidden_dim)
      checker=np.zeros_like(Z1)
      ReLU_Z1 = np.greater(Z1, checker).astype(int) # (batch * hidden_dim)
      Z1 = ReLU_Z1 * Z1
      
      Z1W2 = np.exp(np.matmul(Z1, W2)) # (batch * num_classes)
      norm_Z1W2 = Z1W2 / np.sum(Z1W2, axis=1)[:,None] # =/=
      I_y = np.zeros_like(norm_Z1W2)
      np.put_along_axis(I_y, y_batch[:,None], 1, axis=1)  # (batch x num_classes)
      G2 = norm_Z1W2 - I_y # (batch x num_classes)
      
      G2W2 = np.matmul(G2, W2.T) # (batch x hidden_dim)
      G1 = ReLU_Z1 * G2W2 # (batch x hidden_dim)  

      grad_W1 = np.matmul(X_batch.T, G1) / batch # (input_dim x hidden_dim)
      grad_W2 = np.matmul(Z1.T, G2) / batch # (hidden_dim x num_classes)
      W1 -= lr * grad_W1
      W2 -= lr * grad_W2
      
      # ----
      # ---trial--- new W1 used to compute W2
      # ----
      # Z1 = np.matmul(X_batch, W1) # (batch * hidden_dim)
      # checker=np.zeros_like(Z1)
      # ReLU_Z1 = np.greater(Z1, checker).astype(int) # (batch * hidden_dim)

      # # print(ReLU_Z1)
      # # Z1 = ReLU_Z1
      # # Z1 /= n_iterations*batch

      # Z1W2 = np.exp(np.matmul(Z1, W2)) # (batch * num_classes)
      # norm_Z1W2 = Z1W2 / np.sum(Z1W2, axis=1)[:,None] # =/=
      # # print(norm_Z1W2)
      # I_y = np.zeros_like(norm_Z1W2)
      # np.put_along_axis(I_y, y_batch[:,None], 1, axis=1)  # (batch x num_classes)
      # # print(I_y)
      # G2 = norm_Z1W2 - I_y # (batch x num_classes 
      
      # G2W2 = np.matmul(G2, W2.T) # (batch x hidden_dim)
      # G1 = ReLU_Z1 * G2W2 # (batch x hidden_dim)  
      # # grad_W2 += np.matmul(Z1.T, G2) / batch # (hidden_dim x num_classes)


      
 
    # grad_W1 /= n_iterations
    # grad_W2 /= n_iterations

    # W1 -= lr * grad_W1
    # W2 -= lr * grad_W2


    # print(W1_0 - W1)
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)

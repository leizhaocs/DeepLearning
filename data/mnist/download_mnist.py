import os

if not os.path.exists("train-images-idx3-ubyte"):
    os.system("wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    os.system("wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    os.system("wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    os.system("wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

    os.system("gunzip train-images-idx3-ubyte.gz")
    os.system("gunzip train-labels-idx1-ubyte.gz")
    os.system("gunzip t10k-images-idx3-ubyte.gz")
    os.system("gunzip t10k-labels-idx1-ubyte.gz")
    
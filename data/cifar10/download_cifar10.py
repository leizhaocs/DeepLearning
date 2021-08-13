import os

if not os.path.exists("cifar-10-batches-bin"):
    os.system("wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
    os.system("tar zxvf cifar-10-binary.tar.gz")
    os.system("rm cifar-10-binary.tar.gz")
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

X_train, y_train = load_data('train_32x32.mat')
X_test, y_test = load_data('test_32x32.mat')

print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

print("Training Set", X_train.shape)
print("Test Set", X_test.shape)
print('')

num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)

print(np.unique(y_train))

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

print(np.unique(y_train))

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)

print("Training Set", train_greyscale.shape)
print("Test Set", test_greyscale.shape)
print('')

train_mean = np.mean(train_greyscale, axis=0)

train_std = np.std(train_greyscale, axis=0)

train_greyscale_norm = (train_greyscale - train_mean) / train_std
test_greyscale_norm = (test_greyscale - train_mean)  / train_std

train_greyscale_norm = train_greyscale_norm.astype(np.uint8)
y_train = y_train.astype(np.uint8)
test_greyscale_norm = test_greyscale_norm.astype(np.uint8)
y_test = y_test.astype(np.uint8)

train_data = open("train_data", 'w+b')
train_label = open("train_label", 'w+b')
test_data = open("test_data", 'w+b')
test_label = open("test_label", 'w+b')

train_data.write(np.int32(2051))
train_data.write(np.int32(73257))
train_data.write(np.int32(32))
train_data.write(np.int32(32))

train_label.write(np.int32(2049))
train_label.write(np.int32(73257))

test_data.write(np.int32(2051))
test_data.write(np.int32(26032))
test_data.write(np.int32(32))
test_data.write(np.int32(32))

test_label.write(np.int32(2049))
test_label.write(np.int32(26032))

train_data_binary = bytearray(train_greyscale_norm)
train_label_binary = bytearray(y_train)
test_data_binary = bytearray(test_greyscale_norm)
test_label_binary = bytearray(y_test)
train_data.write(train_data_binary)
train_label.write(train_label_binary)
test_data.write(test_data_binary)
test_label.write(test_label_binary)

train_data.close()
train_label.close()
test_data.close()
test_label.close()

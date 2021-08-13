import numpy as np
import math

np.set_printoptions(suppress=True)

X = [[[0.1, -0.2,  0.3, -0.4,  0.5],
      [0.3,  0.8,  0.3,  0.4,  0.2],
      [0.4,  0.7, -0.3,  0.2,  0.0],
      [0.2,  0.4,  0.6,  0.0,  0.0],
      [0.4, -0.6,  0.8,  0.0, -0.2]],
     [[0.2,  0.2,  0.1,  0.2,  0.7],
      [-0.6, 0.4,  0.2,  0.5,  0.9],
      [0.8, -0.6,  0.3,  0.3,  0.4],
      [0.0,  0.8, -0.2,  0.3, -0.3],
      [-0.4, 0.0,  0.8, -0.2,  0.0]]]
X = np.array(X)

L = [0, 1]
L = np.array(L)

f = [[[0.1, -0.2],
      [-0.3, 0.3]],
     [[0.1, 0.5],
      [0.4, -0.7]]]
f = np.array(f)

bf = 0.2
bf = np.array(bf)

w = [[0.1, -0.2, 0.4, -0.6], [0.0, 0.1, 0.2, 0.6]]
w = np.array(w)

bw = [0.1, -0.2]
bw = np.array(bw)

print("=======================================")
print("Input")
print(X)

print("=======================================")
print("Label")
print(L)

print("=======================================")
print("Conv filter")
print(f)

print("=======================================")
print("Conv bias")
print(bf)

print("=======================================")
print("Full weights")
print(w)

print("=======================================")
print("Full bias")
print(bw)


################################################

conv = np.zeros((4, 4))

for target_x in range(0, 4):
    for target_y in range(0, 4):
        start_x = target_x
        start_y = target_y
        total = 0.2
        for c in range(0, 2):
            for x in range(0, 2):
                for y in range(0, 2):
                    total += X[c, start_x+x, start_y+y] * f[c, x, y]
        conv[target_x, target_y] = total

print("=======================================")
print("Forward conv")
print(conv)

################################################

relu = np.zeros((4, 4))

for i in range(0, 4):
    for j in range(0, 4):
        if conv[i, j] < 0:
            relu[i, j] = 0
        else:
            relu[i, j] = conv[i, j]

print("=======================================")
print("Forward relu")
print(relu)

################################################

pool = np.zeros((2, 2))

for target_y in range(0, 2):
    for target_x in range(0, 2):
        start_x = target_x*2
        start_y = target_y*2
        max_ele = 0
        for y in range(0, 2):
            for x in range(0, 2):
                if max_ele < relu[start_y+y, start_x+x]:
                    max_ele = relu[start_y+y, start_x+x]
        pool[target_y, target_x] = max_ele

print("=======================================")
print("Forward pool")
print(pool)

################################################

pool = pool.reshape((4))
full = np.zeros(2)

for target in range(0, 2):
    full[target] = bw[target]
    for i in range(0, 4):
        full[target] += pool[i] * w[target, i]

print("=======================================")
print("Forward full")
print(full)

################################################

softmax = np.zeros(2)

maxele = full[0]
for i in range(0, 2):
    if maxele < full[i]:
        maxele = full[i]
sumele = 0
for i in range(0, 2):
    softmax[i] = math.exp(full[i] - maxele)
    sumele += softmax[i]
for i in range(0, 2):
    softmax[i] = softmax[i] / sumele

print("=======================================")
print("Forward softmax")
print(softmax)

################################################

loss_softmax = np.zeros(2)

for i in range(0, 2):
    if L[i] == 1:
        loss_softmax[i] = math.log(softmax[i]) * -1
    else:
        loss_softmax[i] = 0

print("=======================================")
print("Backward softmax")
print(loss_softmax)

################################################

loss_full = np.zeros(2)

for i in range(0, 2):
    if loss_softmax[i] != 0:
        loss_full[i] = softmax[i] -1
    else:
        loss_full[i] = softmax[i]

print("=======================================")
print("Backward full")
print(loss_full)

################################################

loss_pool = np.zeros(4)

for i in range(0, 4):
    loss_pool[i] = 0;
    for target in range(0, 2):
        loss_pool[i] += loss_full[target] * w[target, i]

loss_pool = loss_pool.reshape((2,2))

print("=======================================")
print("Backward pool")
print(loss_pool)

################################################

loss_relu = np.zeros((4, 4))

for source_y in range(0, 2):
    for source_x in range(0, 2):
        maxele = -1
        maxx = -1
        maxy = -1
        for target_x in range(source_x*2, source_x*2+2):
            for target_y in range(source_y*2, source_y*2+2):
                if maxele < relu[target_x, target_y]:
                    maxele = relu[target_x, target_y]
                    maxx = target_x
                    maxy = target_y
        loss_relu[maxx, maxy] = loss_pool[source_x, source_y]

print("=======================================")
print("Backward relu")
print(loss_relu)

################################################

loss_conv = np.zeros((4, 4))

for i in range(0, 4):
    for j in range(0, 4):
        if conv[i, j] > 0:
            loss_conv[i, j] = loss_relu[i, j]

print("=======================================")
print("Backward conv")
print(loss_conv)

################################################

loss_x = np.zeros((2, 5, 5))

for target_x in range(0, 4):
    for target_y in range(0, 4):
        start_x = target_x
        start_y = target_y
        for c in range(0, 2):
            for x in range(0, 2):
                for y in range(0, 2):
                    loss_x[c, start_x+x, start_y+y] += loss_conv[target_x, target_y] * f[c, x, y]

print("=======================================")
print("Backward x")
print(loss_x)


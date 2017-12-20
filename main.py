import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0, 1, 1, 0]]).T


Wi = np.random.rand(2,2) - 0.5

b1 = np.random.rand()

Wh = np.random.rand(1, 2) -0.5

b2 = np.random.rand()

def sigm(x):
    return 1/(1 + np.exp(-x))

def d_sigm(x):
    sgm = sigm(x)
    return sgm * (1 - sgm)

def feed_forward(x, Wi, b1, Wh, b2):
    h_in = np.matmul(Wi, x) + b1
    h_out = sigm(h_in)
    o_in = np.matmul(Wh, h_out) + b2
    o_out = sigm(o_in)
    return h_in, h_out, o_in[0], o_out[0]


LR = 0.5

# E - error
# h - hidden layer
# o - output layer
# i/o - input/output

ITERS = 20000
interwal = 100
results = []
for i in range(ITERS):
    outputs = []
    for index in range(len(Y)):
        x = X[index]
        y = Y[index]
        h_in, h_out, o_in, o_out = feed_forward(x, Wi, b1, Wh, b2)

        d_Eo_out = -y + o_out
        d_o_outo_in = d_sigm(o_in)

        d_Eo_in = d_Eo_out * d_o_outo_in

        d_Eb2 = d_Eo_in

        d_Ew5 = d_Eo_in * h_out[0]
        d_Ew6 = d_Eo_in * h_out[1]

        d_h1oi = d_sigm(h_in[0])
        d_h2oi = d_sigm(h_in[1])

        d_Ew1 = d_Eo_in * Wh[0][0] * d_h1oi * x[0]
        d_Ew2 = d_Eo_in * Wh[0][1] * d_h2oi * x[0]
        d_Ew3 = d_Eo_in * Wh[0][0] * d_h1oi * x[1]
        d_Ew4 = d_Eo_in * Wh[0][1] * d_h2oi * x[1]

        d_Eb1 = d_Eo_in * (Wh[0][0] * d_h1oi + Wh[0][1] * d_h2oi)

        b2 -= LR * d_Eb2
        Wh[0][0] -= LR * d_Ew5
        Wh[0][1] -= LR * d_Ew6


        Wi[0][0] -= LR * d_Ew1
        Wi[1][0] -= LR * d_Ew2
        Wi[0][1] -= LR * d_Ew3
        Wi[1][1] -= LR * d_Ew4

        b1 -= LR * d_Eb1
        if i % interwal == 0:
            print('Target: ' + str(y[0]) + ' Got: ' + str(o_out))
            outputs.append(o_out)
    if i % interwal == 0:
        print('---------------------------------')
        results.append(outputs)
        outputs =[]

results = np.array(results)
zz = results[:, 0]
zo = results[:, 1]
oz = results[:, 2]
oo = results[:, 3]

indices = [i * interwal for i in range(int(ITERS/interwal))]

hzz, = plt.plot(indices, zz, label='(0,0)')
hzo, = plt.plot(indices, zo, label='(0,1)')
hoz, = plt.plot(indices, oz, label='(1,0)')
hoo, = plt.plot(indices, oo, label='(1,1)')
plt.legend(handles = [hzz, hzo, hoz, hoo])
plt.show()

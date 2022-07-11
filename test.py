# import numpy as np

# a = np.array([1,3])

# print(np.tile(a, 2))

# def test(a, i, j):
#     print(a, i, j)


# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# arr = np.reshape(arr, (2, 4))

# print(np.sum(arr, axis=0))


# print("{0:06b}".format(13))

import platform
import socket
print(platform.processor())
print(socket.getfqdn())

class QR:
    def __getitem__(self, key):
        print(key, type(key))


qr = QR()
qr[1, 2, 3]

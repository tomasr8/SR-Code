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

    def __setitem__(self, key, value):
        print(key, value)


qr = QR()
qr[1]
qr[1, 3]
qr[1:3]
qr[1, 2:5, 3]
qr[:1, 2:5, 3:, 3:7:2]

# qr[1:3, 's'] = 17

class Dimensions:
    def __init__(self, pixels_per_square):
        self.pixels_per_square = pixels_per_square
        self.squares = 24
        self.small_circle_radius = 2
        self.big_circle_radius = 2
        self.center = self.squares / 2
        self.data_size = self.squares - 2*3

        self._convertible_keys = {
            'squares', 'small_circle_radius', 'big_circle_radius', 'center'}

    def __getattr__(self, key):
        if key.endswith('_px') and key[:-3] in self._convertible_keys:
            key = key[:-3]
            return getattr(self, key) * self.pixels_per_square
        else:
            return getattr(self, key)

dimensions = Dimensions(40)
print(dimensions.center_px)
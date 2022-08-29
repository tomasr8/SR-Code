import numpy as np


CHARACTERS = " !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_SIZE = 6
DUPLICATION_FACTOR = 3


def encode_message(message):
    message = message.ljust(16)
    encoded = []
    for letter in message:
        index = CHARACTERS.find(letter)
        binary = f"{index:06b}"
        encoded += list(map(int, binary))

    # Redundancy!
    return encoded + encoded + encoded


def decode_data(data):
    data = np.reshape(data, (3, len(data) // 3))
    data = np.sum(data, axis=0)
    # Majority vote
    data[data <= 1] = 0
    data[data >= 2] = 1

    # We needed 6 bytes to encode values between 0 and 64
    message_length = len(data) // 6
    data = np.reshape(data, (message_length, 6))
    data = map(binary_array_to_integer, data)
    return "".join([CHARACTERS[n] for n in data])


def binary_array_to_integer(binary):
    string = "".join(map(str, binary))
    return int(string, 2)

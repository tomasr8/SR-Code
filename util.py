from xml.dom.pulldom import CHARACTERS
import numpy as np

# Just so happens to be 64 (2^6) characters
CHARACTERS = " !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def encode_string(message):
    assert len(message) <= 16, "Maximum length we can encode is 16 characters"
    assert all(letter in CHARACTERS for letter in message), f"The source string needs to use characters from this set: {CHARACTERS}"

    message = message.ljust(16)
    encoded = []
    for letter in message:
        index = CHARACTERS.find(letter)
        binary = f"{index:06b}"
        encoded += list(binary)

    # Redundancy!
    return encoded + encoded + encoded


def decode_data(data):
    data = np.reshape(data, (3, len(data)//3))
    data = np.sum(data, axis=0)
    # Majority vote
    data[data <= 1] = 0
    data[data >= 2] = 1

    # We needed 6 bytes to encode values between 0 and 64
    message_length = len(data)//6
    data = np.reshape(data, (message_length, 6))
    data = map(binary_array_to_integer, data)
    return "".join([CHARACTERS[n] for n in data])


def binary_array_to_integer(binary):
    string = "".join(map(str, binary))
    return int(string, 2)

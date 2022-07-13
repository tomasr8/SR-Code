from util import encode_string, decode_data


msg = "Hello world!"


print(decode_data(encode_string(msg)))

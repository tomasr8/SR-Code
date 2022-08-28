<p align="center">
  <img src="https://raw.githubusercontent.com/tomasr8/SR-Code/main/assets/logo.svg">
</p>

# - A simplified `QR code`-like reader and generator

![](example.mp4)
<video src="example.mp4" />

To see more examples, check ..

## What is this?
This project is not trying to be a serious contender to a QR code. It is just a hobby project that I work on in my free time that I wanted to make available for others. I tried to keep the code simple and clear so that people not well versed in computer vision (me included) can experiment with it and learn from.

SR is a tongue-in-cheek acronym for `Sufficient Response`, since QR stands for `Quick Response` (atleast that's what wikipedia tells me).

## How do I use this?

It's available via pip:

```bash
pip install srcode
```

### As a CLI tool

```bash
# Create a 600x600 SR code which encodes 'Hello world!'
sr generate -s 600 -m 'Hello world!' hello.png
# Decode an SR code from an image
sr decode hello.png
# Decode from a video
# Press 'b' to toggle between color and black&white mode
sr video video.mp4
# Alternatively use the system camera
sr video
```

Check the help message of the individual commands e.g. `sr generate --help` for more info.

### As a library

Import the relevant functions:

```python
import cv2
from sr import generate, decode, decode_video

# Generating an image
image = generate(size=400, "Test image")
cv2.imwrite(image, "test.png")

# Decoding an image
image = cv2.imread("test.png")
# Expectes a 'BGR' image (opencv's default)
result = decode(image, find_all=True, use_bw=False)
# Result is an instance of 'DecodeResult'
print(result.success)
# Typically multiple candidate contours are found.
# Only those which are successfully read will have a message.
for contour in result.contours:
  if contour.success:
    print(f"Message: {contour.message}")
  else:
    print(f"Error: {contour.error}")


# Decode from a video
# Pass '0' to open the default camera stream
decode_video(filename, visualize=True, verbose=True, stop_on_success=False)
```

For more info, check how the CLI uses these functions in [sr/sr.py](sr/sr.py)

## Limitations

Before I get into how it works, here are the limitations of the SR code which stem from the fact of trying to keep the design simple
while still having some built-in robustness.

- Maximum message length of 16 characters - the SR code uses a simple error correction mechanism which takes space.
- The characters must be from this set: ` !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`. It is not a coincidence that the set has 64 characters in it.
- The SR code cannot handle mirrored images - This could be easily remedied, however at a cost of added complexity.


## How does it work?
if you wanna see a detailed explanation, skip to [Design](#design).

![](https://raw.githubusercontent.com/tomasr8/SR-Code/main/assets/sr-diagram.png)

This image above is a diagram showing all the main parts of the SR code. I'll go over them one by one in the following section. (As a side note, even with the extra colored lines, this diagram will decode correctly. Try running `sr decode assets/sr-diagram.png` to see for yourself)

Let's assume we start with a standard RGB image that includes the SR code somewhere in it. Before we can attempt to decode it, we must first locate it.

This is done by finding the outer border shown in the diagram above. The border is nicely sandwiched between black and white regions which together define a [contour](https://learnopencv.com/contour-detection-using-opencv-python-c/) - basically a closed shape separating two regions of an image. OpenCV has some fancy algorithms
which can find these contours.

Running the contour finding algorithm, we end up with something like this:

<p align="center">
  <img src="contours.png">
</p>

You can see there are multiple contours (shown in orange).

Once we have a candidate contour, we apply [perspective correction](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html). We need this in case the is seen under an angle. This makes as if we were looking directly at it.

- image

Now that we have a nice flat image, we check for the presence of the inner rings, again shown in the diagram above. This is just to make sure that we don't treat any random contour as a valid code. A lot of everyday objects can give false positives here. On the other hand, if we find the rings, it's likely we actually have a real SR code.


Having this, we locate the start corner. The start corner tells where to start reading the data:

- image

Finally, we read the data. A black square encodes the bit 1 and white encodes 0.

- image


## Design

### Encoding

The SR code can encode up to 16 characters from a slightly modified base64 character set: ` !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`. The difference is that instead of '+/' we have `space` and `!` which I think are a bit more useful for encoding text messages.

Since the character set size is 64, any character can be encoded with 6 bits (since `2^6=64`). To offer some error correction capabilities, the message is duplicated 3 times. When decoding, a majority vote is used which gives correct results as long as at least two copies of each bit of the message are correct. Since we have 288 total available data squares in the image, this gives us a maximum message length of 16, seeing that `288/(6*3) = 16`.

This encoding scheme is pretty inefficient and it is possible to use more sophisticated error correction algorithms perhaps combined with some compression which would allow for longer messages to fit.

### Thresholding

The OpenCV contour-finding algorithm only work with black&white. Using a static threshold to create a black&white image is very sensitive to lighting conditions in which the image was taken. Instead, we use [Otsu's Binarization](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) which selects the treshold dynamically using the image histogram. This gives a more robust output and it's one parameter less to tune.

### Perspective correction

Since any random image of an SR code is unlikely to be perfectly facing the camera. This is a problem for decoding because the individual squares won't line up along the image x and y axes. Before attempting to decode it, we must first undistort the image. To achieve this, we apply a homography estimated from the contour. This gives us the undistorted QR code.

### Reading the data

After the perspective correction, we check for presence of the inner rings.
As I explained previously, this is to reject false positives. The reason for using the rings is that it is a pretty specific shape which is unlikely to appear randomly. It is also symmetrical so we can check for it regardless of the orientation the code.

Once we have found the start corner, we rotate the code so that the start corner is in the top left. Then, we simply read and decode the data from top to bottom and column by column.

## Misc

### Tab completion

Click includes autocomplete capability, just run this command in your `bash` shell:

```bash
eval "$(_SR_COMPLETE=bash_source sr)"
```

Or in `zsh` (you might need to run `rehash` first)

```bash
eval "$(_SR_COMPLETE=zsh_source sr)"
```

### Further reading

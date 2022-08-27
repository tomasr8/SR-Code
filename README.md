<p align="center">
  <img src="https://raw.githubusercontent.com/tomasr8/SR-Code/main/assets/logo.svg">
</p>

# A simplified QR code

## But why?

I wanted to do a project related to computer vision but I didn't want to create yet-another-QR-code-reader™️ but rather create something of my own.
This project is not trying to be a serious contender to a QR code. It is just a hobby project that I work on in my free time. I tried to keep the code simple and clear so that
even people with little experience in computer vision can experiment with it easily.


SR is a tongue-in-cheek acronym for `Sufficient Response`, since QR stands for `Quick Response` (atleast that's what wikipedia tells me).

## Examples

## How do I use this?

Clone the repo, create a virtual environment and install as a dev package:

```bash
git clone --depth=1 https://github.com/tomasr8/SR-Code.git
cd SR-Code

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

To use it:

```bash
# Create a 600x600 SR code which encodes 'Hello world!'
sr generate -s 600 -m 'Hello world!' hello.png
# Decode an SR code from an image
sr decode hello.png
# Decode from a video
sr video video.mp4
# Alternatively use the system camera
sr video
```

For help:

```bash
sr --help
```

## How does it actually work?
if you wanna see a detailed explanation, skip to [Design](#design).

![](https://raw.githubusercontent.com/tomasr8/SR-Code/main/assets/sr-diagram.png)

This is a diagram showing all the main parts of the SR code. We'll explain them in the following section. (By the way, even with the extra colored lines, this will still decode correctly. Try running `sr decode sr-diagram.png` to see for yourself)

Let's assume we start with a standard RGB image that includes the code somewhere inside of it. Before we even attempt to decode it we must first locate it.

This is done by finding the outer border shown in the diagram above. The border is nicely sandwiched between black and white regions which together define a [contour](https://learnopencv.com/contour-detection-using-opencv-python-c/). OpenCV has a some fancy algorithms
which can find these contours.

We first threshold the image to convert to a black & white and then run OpenCV's `findContours`.

We end up with something like this:


Once we have a candidate contour, we apply [perspective correction](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html). We need this in case the is seen under an angle. This makes as if we were looking directly at it.

- image

Now that we have a nice flat image, we check for the presence of the inner rings, again shown in the diagram above. This is just to make sure that we don't treat any random contour as a QR code. A lot of everyday objects can give false positives here. However, if we find the rings, it's likely we actually a real QR code.


Having this, we locate the start corner. The start corner tells where to start reading the data:

- image

Finally, we read the data. A black square encodes the bit 1 and white encodes 0.

- image


## Design

### Encoding

The SR code can encode up to 16 characters from a slightly modified base64 character set: ` !0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`. The difference is that instead of '+/' we have `space` and `!` which I think are a bit more useful for encoding text messages.

Since the character set size is 64, any character can be encoded with 6 bits (since 2^6=64). To offer some error correction capabilities the message is duplicated 3 times. When decoding a majority vote is used which will give results as long as at least two copies are correct. Since we have 288 total available data squares, this gives us a maximum message length of 16, seeing that 288/(6*3) = 16.

This encoding scheme is pretty inefficient and it's in principle possible to use more sophisticated encoding algorithms which would let you fit longer messages.

### Thresholding

To be able to find the contours, the image must be first converted to black & white.
Using a static threshold is sensitive to lighting conditions. Instead, we use [Otsu's Binarization](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) which selects the treshold dynamically using the image histogram.


### Perspective correction

Since any random image of a QR code is unlikely to be perfectly facing the camera, before attempting to decode it, we must first undistort the image. To achieve this, we apply a homography estimated from the contour. This gives us the undistorted QR code.

### Reading the data

After the perspective correction, we check for presence of the inner rings.
As I explained previously, this is to reject false positives. The reason for using the rings is that it is a pretty specific shape which is unlikely to appear randomly. It is also symmetrical so we can check for it regardless of the orientation the code.

Once we have found the start corner, we rotate the code so that the start corner is in the top left. Then, we simply read and decode the data from top to bottom and column by column.


## As a library

You can import the relevant functions:

```python
from sr import generate, decode, decode_video, EncodeError

image = generate(size=400, "Test image")


```

## Misc

### Tab completion

Click includes autocomplete capability, just run this command in your shell:
```bash
eval "$(_SR_COMPLETE=zsh_source sr)"
```

for zsh (you might need to run `rehash` as well), or for bash:
```bash
eval "$(_SR_COMPLETE=bash_source sr)"
```

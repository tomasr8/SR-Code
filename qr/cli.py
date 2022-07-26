import click
import cv2

from qr.code import SRCodeGenerator
from qr.decode import decode, decode_video


@click.group()
def cli():
    pass


@cli.command()
@click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
@click.option("--message", "-m", type=str, required=True, help="The message to encode")
@click.argument("output_file")
def generate(size, message, output_file):
    qr = SRCodeGenerator(pixels_per_square=size)

    if error := qr.validate_message(message):
        click.secho(error, fg='red')
        return

    image = qr.generate(message)
    cv2.imwrite(output_file, image)


@cli.command()
@click.argument("input_file")
def read(input_file):
    image = cv2.imread(input_file)
    result = decode(image)
    print(result)

    cv2.imshow('Frame', result.bw_visualization)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.imshow('Frame', result.contour_visualization)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.imshow('Frame', result.decode_visualization)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


@cli.command()
@click.argument("input_file")
def video(input_file):
    decode_video(0)

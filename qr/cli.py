import click
import cv2

from qr.generate_qr import generate_qr

@click.group()
def cli():
    pass

@cli.command()
@click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
@click.option("--message", "-m", type=str, required=True, help="The message to encode")
@click.argument("output_file")
def generate(size, message, output_file):
    image = generate_qr(message, size)
    cv2.imwrite(output_file, image)


@cli.command()
# @click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
# @click.option("--message", "-m", type=str, required=True, help="The message to encode")
# @click.argument("output_file")
def read():
    raise NotImplementedError("Read method not implemented")

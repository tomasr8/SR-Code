import click
import cv2

from qr.generate_qr import QrCodeGenerator


@click.group()
def cli():
    pass


@cli.command()
@click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
@click.option("--message", "-m", type=str, required=True, help="The message to encode")
@click.argument("output_file")
def generate(size, message, output_file):
    qr = QrCodeGenerator(pixels_per_square=size)

    if error := qr.validate_message(message):
        click.secho(error, fg='red')
        return

    image = qr.generate(message)
    cv2.imwrite(output_file, image)


@cli.command()
# @click.option("--size", "-s", type=int, default=25, help="Size of one square in pixels")
# @click.option("--message", "-m", type=str, required=True, help="The message to encode")
# @click.argument("output_file")
def read():
    raise NotImplementedError("Read method not implemented")

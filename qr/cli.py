import click
import cv2

from qr.generate_qr import QrCodeGenerator
from qr.read_qr import read_qr


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
@click.argument("input_file")
def read(input_file):
    image = cv2.imread(input_file)
    read_qr(image)

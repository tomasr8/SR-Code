import click
import cv2

from qr.code import EncodeError, SRCodeGenerator
from qr.decode import decode as _decode
from qr.decode import decode_video
from qr.image import horizontal_concat


@click.group()
def cli():
    pass


@cli.command()
@click.option("--size", "-s", type=int, default=600, help="Size in pixels")
@click.option("--message", "-m", type=str, required=True, help="The message to encode")
@click.argument("output_file")
def generate(size, message, output_file):
    sr = SRCodeGenerator(size=size)

    try:
        image = sr.generate(message)
    except EncodeError as e:
        click.secho(e, fg='red')
    else:
        cv2.imwrite(output_file, image)


@cli.command()
@click.option("--visualization/--no-visualization", default=True, help="Show/hide openCV visualization.")
@click.argument("input_file", type=click.File())
def decode(input_file, visualization):
    image = cv2.imread(input_file.name)
    result = _decode(image)
    click.echo(result)

    if not visualization:
        return

    cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)
    if result.message:
        visualization = horizontal_concat(result.contour_visualization, result.decode_visualization)
    else:
        visualization = result.contour_visualization

    cv2.imshow('Frame', visualization)
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


@cli.command()
@click.argument("input_file")
def video(input_file):
    decode_video(0)

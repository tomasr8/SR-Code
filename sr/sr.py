import click
import cv2

from sr.code import EncodeError, SRCodeGenerator
from sr.decode import decode as _decode
from sr.decode import decode_video
from sr.image import horizontal_concat, pressed_quit


@click.group()
def cli():
    pass


@cli.command(name="generate")
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


@cli.command(name="decode")
@click.option("--visualize/--no-visualize", default=True, help="Show/hide OpenCV visualization.")
@click.option("--find-all/--no-find-all", default=True, help="Find all SR codes.")
@click.argument("input_file", type=click.File())
def decode(input_file, visualize, find_all):
    image = cv2.imread(input_file.name)
    result = _decode(image, find_all)
    click.echo(result)

    if not visualize:
        return

    cv2.namedWindow('SR code', cv2.WINDOW_KEEPRATIO)
    visualization = result.contour_visualization
    if result.success:
        for vis in result.get_visualizations():
            visualization = horizontal_concat(visualization, vis)

    cv2.imshow('SR code', visualization)
    while not pressed_quit():
        pass

    cv2.destroyAllWindows()


@cli.command(name="video")
@click.option("--visualize/--no-visualize", default=True, help="Show/hide OpenCV visualization.")
@click.option("--verbose/--no-verbose", default=True, help="Print decode result for every frame.")
@click.option("--stop-on-success/--no-stop-on-success", default=False, help="Stop on a successful read.")
@click.argument("input_file", type=click.File(), required=False)
def video(input_file, visualize, verbose, stop_on_success):
    # '0' signals OpenCV to open a camera stream
    filename = input_file.name if input_file else 0
    decode_video(filename, visualize, verbose, stop_on_success)

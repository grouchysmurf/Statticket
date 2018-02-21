import argparse
from enum import Enum
import io
import os

from google.cloud import vision
from google.cloud.vision import types

from PIL import Image, ImageDraw


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

class BreakType(Enum):
    UNKNOWN	= 0
    SPACE = 1
    SURE_SPACE = 2
    EOL_SURE_SPACE = 3
    LINE_BREAK = 4
    HYPHEN = 5


def get_response(image_file):
    """Returns the full text annotation and properties given an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response_text = client.document_text_detection(image=image)
    document = response_text.full_text_annotation
    response_props = client.image_properties(image=image)
    props = response_props.image_properties_annotation

    return document, props


def get_document_bounds(document, feature):
    """Returns document bounds given an image."""

    bounds = []

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)

                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

        if (feature == FeatureType.PAGE):
            bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image


def render_doc_text(image_file, document, img_out):
    
    image = Image.open(image_file)

    bounds = get_document_bounds(document, FeatureType.PAGE)
    draw_boxes(image, bounds, 'blue')
    bounds = get_document_bounds(document, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    bounds = get_document_bounds(document, FeatureType.WORD)
    draw_boxes(image, bounds, 'yellow')

    if img_out is not 'no':
        outfile_name = os.path.join(os.path.dirname(image_file),os.path.splitext(image_file)[0] + '_out' + os.path.splitext(image_file)[1])
        image.save(outfile_name)
    else:
        pass

def print_doc_text(document):
    for page in document.pages:
        for block in page.blocks:
            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)

            block_symbols = []
            block_text = ''
            for word in block_words:
                block_symbols.extend(word.symbols)
                for symbol in block_symbols:
                    block_text = block_text + symbol.text
                    if symbol.property.detected_break.type > 0 and symbol.property.detected_break.type < 5:
                        block_text = block_text + ' '
                block_symbols = []

            print('Block Content: {}'.format(block_text))

    print('Entire detected text: {}'.format(document.text))

def print_doc_props(props):
    print('Properties:')

    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('-img_out', help='Optional, output image with identified text area bounds in output folder', default='no')
    args = parser.parse_args()

    parser = argparse.ArgumentParser()
    document, props = get_response(args.detect_file)
    render_doc_text(args.detect_file, document, args.img_out)
    print_doc_text(document)
    print_doc_props(props)

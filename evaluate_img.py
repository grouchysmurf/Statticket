import argparse
from enum import Enum
import io
import os
import csv
from collections import deque

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

def get_doc_text(document):

    n_blocks = 0
    n_words = 0

    img_text = ''

    for page in document.pages:
        for block in page.blocks:
            n_blocks += 1
            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)

            block_symbols = []
            block_text = ''
            for word in block_words:
                n_words += 1
                block_symbols.extend(word.symbols)
                for symbol in block_symbols:
                    block_text = block_text + symbol.text
                    if symbol.property.detected_break.type > 0 and symbol.property.detected_break.type < 5:
                        block_text = block_text + ' '
                block_symbols = []

            img_text = img_text + block_text + "\n"

    return img_text, n_blocks, n_words

def get_doc_props(props):
    
    img_frax = []

    for color in props.dominant_colors.colors:
        fraction = color.pixel_fraction
        r = color.color.red
        g = color.color.green
        b = color.color.blue
        a = color.color.alpha
        if a == None:
            img_frax.append([fraction,[r,g,b,a]])
        else:
            img_frax.append([fraction,[r,g,b]])
    
    sorted_img_frax = sorted(img_frax, key=lambda img_frax: img_frax[0], reverse=True)

    while len(sorted_img_frax) > 5:
        sorted_img_frax.pop()
    
    return sorted_img_frax[0][0], sorted_img_frax[0][1], sorted_img_frax[1][0], sorted_img_frax[1][1], sorted_img_frax[2][0], sorted_img_frax[2][1], sorted_img_frax[3][0], sorted_img_frax[3][1], sorted_img_frax[4][0], sorted_img_frax[4][1]


def make_csv(csv_lines):

    title_lines = ['Image', 'text', 'blocks', 'words', 'frax_1', 'rgb_1', 'frax_2', 'rgb_2', 'frax_3', 'rgb_3', 'frax_4', 'rgb_4', 'frax_5', 'rgb_5']
    
    with open('output.csv', mode='w', newline='', encoding='utf-8', errors='strict') as output_file:
        writer = csv.writer(output_file, dialect='excel', delimiter=';')
        writer.writerow(title_lines)
        for line in csv_lines:
            img_file = line[0]
            img_text = line[1]
            n_blocks = line[2]
            n_words = line[3]
            frax_1 = line[4]
            rgb_1 = line[5]
            frax_2 = line[6]
            rgb_2 = line[7]
            frax_3 = line[8]
            rgb_3 = line[9]
            frax_4 = line[10]
            rgb_4 = line[11]
            frax_5 = line[12]
            rgb_5 = line[13]
            writer.writerow([os.path.basename(img_file), img_text, n_blocks, n_words, frax_1, rgb_1, frax_2, rgb_2, frax_3, rgb_3, frax_4, rgb_4, frax_5, rgb_5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_dir', help='Directory containing image files')
    parser.add_argument('-img_out', help='Optional, output image with identified text area bounds in resource folder', default='no')
    args = parser.parse_args()

    parser = argparse.ArgumentParser()

    csv_lines = []

    dir_entries = os.scandir(args.detect_dir)
    for dir_entry in dir_entries:
        if dir_entry.is_file() and dir_entry.name.endswith(".jpg"):
            if dir_entry.name.endswith("_out.jpg"):
                pass
            else:
                document, props = get_response(os.path.abspath(dir_entry))
                render_doc_text(os.path.abspath(dir_entry), document, args.img_out)
                img_text, n_blocks, n_words = get_doc_text(document)
                frax_1, rgb_1, frax_2, rgb_2, frax_3, rgb_3, frax_4, rgb_4, frax_5, rgb_5 = get_doc_props(props)
                csv_lines.append([os.path.abspath(dir_entry), img_text, n_blocks, n_words, frax_1, rgb_1, frax_2, rgb_2, frax_3, rgb_3, frax_4, rgb_4, frax_5, rgb_5])

    make_csv(csv_lines)

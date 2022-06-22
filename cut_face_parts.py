import os
import numpy as np
import json
from PIL import Image, ImageDraw
import random

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

IMAGES_PATH = os.path.join(os.getcwd(), 'images')
FACES_PATH = os.path.join(IMAGES_PATH, 'face_container')
LEFT_EYE_PATH = os.path.join(IMAGES_PATH, 'left_eye')
RIGHT_EYE_PATH = os.path.join(IMAGES_PATH, 'right_eye')
NOSE_PATH = os.path.join(IMAGES_PATH, 'nose')
MOUTH_PATH = os.path.join(IMAGES_PATH, 'mouth')


def _get_polygon_mask(center, width, height):
    top_left = ((center[0] - int(width / 2)) + random.randint(0, int(width / 4)),
                (center[1] - int(height / 2)) + random.randint(0, int(height / 4)))
    top_right = ((center[0] + int(width / 2)) - random.randint(0, int(width / 4)),
                 (center[1] - int(height / 2)) + random.randint(0, int(height / 4)))
    bottom_left = ((center[0] - int(width / 2)) + random.randint(0, int(width / 4)),
                   (center[1] + int(height / 2)) - random.randint(0, int(height / 4)))
    bottom_right = ((center[0] + int(width / 2)) - random.randint(0, int(width / 4)),
                    (center[1] + int(height / 2)) - random.randint(0, int(height / 4)))

    polygon = [top_left, top_right, bottom_right, bottom_left]
    mask_image = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(polygon, outline=255, fill=255)
    return mask_image


def _make_even(value):
    return value if value % 2 == 0 else value - 1


def crop_piece(image, center, width, height, y_offset=0):
    image.convert('RGBA')

    width = _make_even(width)
    height = _make_even(height)

    left = center[0] - int(width / 2)
    upper = center[1] - int(height / 2) - y_offset
    right = center[0] + int(width / 2)
    lower = center[1] + int(height / 2) - y_offset

    mask = _get_polygon_mask((width // 2, height // 2), width, height)
    cropped = image.crop((left, upper, right, lower))

    cropped.putalpha(mask)
    return cropped


def _get_face_image(path):
    image = Image.open(path)
    image = image.convert('RGBA')
    return image


def _denormalize_keypoint(keypoint, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, x_offset=0, y_offset=0):
    return int(keypoint['x'] * image_width) - x_offset, int(keypoint['y'] * image_height) - y_offset


def _denormalize_value(value, size=IMAGE_WIDTH):
    return int(value * size)


def main():
    print('Cropping face parts...')

    with open('face_keypoints.json', 'r') as f:
        keypoints = json.load(f)

        for face in keypoints['faces']:
            print('Processing face {}...'.format(face['file']))

            eye_width = 0.45 * face['bounding_box']['width']
            eye_height = 0.4 * face['bounding_box']['height']
            nose_width = 0.25 * face['bounding_box']['width']
            nose_height = 0.5 * face['bounding_box']['height']
            mouth_width = 0.55 * face['bounding_box']['width']
            mouth_height = 0.3 * face['bounding_box']['height']

            img = _get_face_image(os.path.join(FACES_PATH, face['file']))
            left_eye = crop_piece(img, _denormalize_keypoint(face['left_eye']), _denormalize_value(eye_width),
                                  _denormalize_value(eye_height))
            right_eye = crop_piece(img, _denormalize_keypoint(face['right_eye']), _denormalize_value(eye_width),
                                   _denormalize_value(eye_height))
            nose = crop_piece(img, _denormalize_keypoint(face['nose']), _denormalize_value(nose_width),
                              _denormalize_value(nose_height), y_offset=_denormalize_value(0.05))
            mouth = crop_piece(img, _denormalize_keypoint(face['mouth']), _denormalize_value(mouth_width),
                               _denormalize_value(mouth_height))

            print('Saving images...')
            left_eye.save(os.path.join(LEFT_EYE_PATH, face['file']), 'PNG')
            right_eye.save(os.path.join(RIGHT_EYE_PATH, face['file']), 'PNG')
            nose.save(os.path.join(NOSE_PATH, face['file']), 'PNG')
            mouth.save(os.path.join(MOUTH_PATH, face['file']), 'PNG')

        print('Done!')


if __name__ == '__main__':
    main()

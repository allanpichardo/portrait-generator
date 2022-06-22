import json
import os
from PIL import Image, ImageDraw
import random
from cut_face_parts import _get_face_image, _denormalize_keypoint, _denormalize_value

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

IMAGES_PATH = os.path.join(os.getcwd(), 'images')
FACES_PATH = os.path.join(IMAGES_PATH, 'face_container')
LEFT_EYE_PATH = os.path.join(IMAGES_PATH, 'left_eye')
RIGHT_EYE_PATH = os.path.join(IMAGES_PATH, 'right_eye')
NOSE_PATH = os.path.join(IMAGES_PATH, 'nose')
MOUTH_PATH = os.path.join(IMAGES_PATH, 'mouth')

COLLAGES_PATH = os.path.join(os.getcwd(), 'collages')


def create_directories():
    if not os.path.exists(COLLAGES_PATH):
        os.mkdir(COLLAGES_PATH)
    if not os.path.exists(IMAGES_PATH):
        os.mkdir(IMAGES_PATH)
    if not os.path.exists(FACES_PATH):
        os.mkdir(FACES_PATH)
    if not os.path.exists(LEFT_EYE_PATH):
        os.mkdir(LEFT_EYE_PATH)
    if not os.path.exists(RIGHT_EYE_PATH):
        os.mkdir(RIGHT_EYE_PATH)
    if not os.path.exists(NOSE_PATH):
        os.mkdir(NOSE_PATH)
    if not os.path.exists(MOUTH_PATH):
        os.mkdir(MOUTH_PATH)


def generate_collage(name, use_background=False, color='black'):
    with open('face_keypoints.json', 'r') as f:
        keypoints = json.load(f)

        # create blank image
        img = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT)) if not use_background else Image.new('RGBA', (
        IMAGE_WIDTH, IMAGE_HEIGHT), color)
        img.putalpha(255)

        face = keypoints['faces'][random.randint(0, len(keypoints['faces']) - 1)]

        left_eye_modifier = random.uniform(1.0, 3.0)
        right_eye_modifier = random.uniform(1.0, 3.0)
        nose_modifier = random.uniform(0.8, 1.5)
        mouth_modifier = random.uniform(0.8, 2.2)

        left_eye_width = _denormalize_value(0.25 * face['bounding_box']['width'] * left_eye_modifier)
        left_eye_height = _denormalize_value(0.2 * face['bounding_box']['height'] * left_eye_modifier)
        right_eye_width = _denormalize_value(0.25 * face['bounding_box']['width'] * right_eye_modifier)
        right_eye_height = _denormalize_value(0.2 * face['bounding_box']['height'] * right_eye_modifier)
        nose_width = _denormalize_value(0.25 * face['bounding_box']['width'] * nose_modifier)
        nose_height = _denormalize_value(0.5 * face['bounding_box']['height'] * nose_modifier)
        mouth_width = _denormalize_value(0.5 * face['bounding_box']['width'] * mouth_modifier)
        mouth_height = _denormalize_value(0.25 * face['bounding_box']['height'] * mouth_modifier)

        face_image = _get_face_image(os.path.join(FACES_PATH, face['file']))
        face_image = face_image.convert('RGBA')

        # paste face image
        img.paste(face_image, (0, 0), face_image)

        # load random parts
        left_eye_image = Image.open(os.path.join(LEFT_EYE_PATH, random.choice(os.listdir(LEFT_EYE_PATH)))).resize(
            (left_eye_width, left_eye_height), resample=Image.BICUBIC)
        right_eye_image = Image.open(os.path.join(RIGHT_EYE_PATH, random.choice(os.listdir(RIGHT_EYE_PATH)))).resize(
            (right_eye_width, right_eye_height), resample=Image.BICUBIC)
        nose_image = Image.open(os.path.join(NOSE_PATH, random.choice(os.listdir(NOSE_PATH)))).resize(
            (nose_width, nose_height), resample=Image.BICUBIC)
        mouth_image = Image.open(os.path.join(MOUTH_PATH, random.choice(os.listdir(MOUTH_PATH)))).resize(
            (mouth_width, mouth_height), resample=Image.BICUBIC)

        # paste parts
        img.paste(nose_image, _denormalize_keypoint(face['nose'], x_offset=nose_image.width // 2,
                                                    y_offset=nose_image.height // 2), nose_image)
        img.paste(left_eye_image, _denormalize_keypoint(face['left_eye'], x_offset=left_eye_image.width // 2,
                                                        y_offset=left_eye_image.height // 2), left_eye_image)
        img.paste(right_eye_image, _denormalize_keypoint(face['right_eye'], x_offset=right_eye_image.width // 2,
                                                         y_offset=right_eye_image.height // 2), right_eye_image)
        img.paste(mouth_image, _denormalize_keypoint(face['mouth'], x_offset=mouth_image.width // 2,
                                                     y_offset=mouth_image.height // 2), mouth_image)

        # save image
        img.save(os.path.join(COLLAGES_PATH, "{0:05d}.png").format(name), 'PNG')


def main():
    create_directories()

    collage_count = int(input('How many collages do you want to generate? '))
    use_background = input('Do you want to use a background image? (y/n) ')

    color = 'black'
    if use_background == 'y':
        color = input('Enter a color hex code [black] ')
        color = color if color != '' else 'black'

    print('Generating collages...')
    for i in range(collage_count):
        print('Generating collage {}...'.format(i + 1))
        generate_collage(i, use_background=use_background == 'y', color=color)


if __name__ == '__main__':
    main()

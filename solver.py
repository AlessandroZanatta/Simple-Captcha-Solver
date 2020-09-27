#!/usr/bin/env python3
import numpy as np
import requests
from pwn import *   
import shutil
import os
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from time import sleep, time
import pytesseract
from multiprocessing import Process, Manager
from ocr import predict

WHITE = 255
BLACK = 0
LETTERS_IN_IMAGE = 6
DEBUG = False

# sensibility for binary transformation
LOWER_THRESHOLD = 170
UPPER_THRESHOLD = 255

# connection params & session
base_url = "http://reversecaptcha.hax.w3challs.com"
s = requests.Session()

def start():
    """
        Clear images folder and start session 
    """
    global s
    global base_url

    log.info("Clearing images folder...")
    for filename in os.listdir("images/"):
        os.unlink("images/" + filename)

    log.info("Starting session...")
    s.get(base_url)


def get_image(index):
    """
        Retrieve index-th image
    """
    global s
    global base_url
    
    log.info(f"Getting image -> {index}")
    url = f"{base_url}/gen_captcha.php?num={index}&"
    r = s.get(url, stream=True)
    r.raw.decode_content = True
    path = f"images/captcha{index}.png"
    with open(path,'wb') as f:
        log.success(f"Downloaded image @ {path}")
        shutil.copyfileobj(r.raw, f)


def to_binary(index):
    """
        Transforms index-th image into a binary image (with all pixels to 0 or 255)
        It uses global variables LOWER_THRESHOLD and UPPER_THRESHOLD as thresholds for the binarization
    """
    log.info(f"Detecting blobs in captcha {index}")
    path = f"images/captcha{index}.png"

    # Import and turn into grayscale
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # To binary and plot
    _, binary = cv2.threshold(gray, LOWER_THRESHOLD, UPPER_THRESHOLD, cv2.THRESH_BINARY_INV)
    
    # plt.imshow(binary, cmap="gray")
    # plt.show()

    Image.fromarray(binary).save(f"images/captcha_binary_{index}.png")

# ==================== UTILITY FUNCTIONS ==================== #


def get_ascii_image(matrix, w=120):

    """
        Prints given matrix as an ascii image
    """

    img = Image.fromarray(matrix)

    # resize the image
    width, height = img.size
    aspect_ratio = height/width
    new_width = w
    new_height = aspect_ratio * new_width * 0.55
    img = img.resize((new_width, int(new_height)))
    # new size of image
    # print(img.size)

    # convert image to greyscale format
    img = img.convert('L')

    pixels = img.getdata()

    # replace each pixel with a character from array
    chars = ["B","S","#","&","@","$","%","*","!",":","."]
    new_pixels = [chars[pixel//25] for pixel in pixels]
    new_pixels = ''.join(new_pixels)

    # split string of chars into multiple strings of length equal to new width and create a list
    new_pixels_count = len(new_pixels)
    ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
    ascii_image = "\n".join(ascii_image)

    print(ascii_image)
    print()
    sys.stdout.flush()

def cut_left_border(image, max_x):

    image = image.transpose()

    x = 0
    while x < max_x and not any(image[x] == WHITE):
        x += 1
    
    return image[x:].transpose()

def cut_right_border(image, max_x):

    image = image.transpose()

    x = max_x
    while x > 0 and not any(image[x] == WHITE):
        x -= 1
    
    return image[:x].transpose()


def cut_upper_border(image, max_y):

    y = 0
    while y < max_y and not any(image[y] == WHITE):
        y += 1
    
    return image[y:]

def cut_lower_border(image, max_y):

    y = max_y
    while y > 0 and not any(image[y] == WHITE):
        y -= 1
    
    return image[:y+1]


def max_x(arr):

    max_ = 0
    for _, x in arr:
        if x > max_:
            max_ = x

    return max_


def min_x(arr):
    min_ = 200
    for _, x in arr:
        if x < min_:
            min_ = x

    return min_


def copy_bits_left(y, x, src, dest, start=0):

    for i in range(start, x):
        dest[y][i] = src[y][i]


def copy_bits_right(y, x, src, dest, stop):

    for i in range(x+1, stop+1):
        dest[y][i] = src[y][i]


def cut_upper_and_lower(image):

    height, _ = image.shape
    image = cut_upper_border(image, height-1)

    height, _ = image.shape
    image = cut_lower_border(image, height-1)
    return image



def pad(letter, final_width, final_height):
    """
        Normalizes the given letter size padding it into a final_width x final_height image and returns it
    """
    curr_height, curr_width = letter.shape

    pad_width = int((final_width - curr_width) / 2)
    pad_height = int((final_height - curr_height) / 2)


    letter = np.pad(letter, [(pad_height, ),(pad_width, )], mode='constant', constant_values=BLACK)
    
    # account for odd height/width
    letter = np.pad(letter, [(0, curr_height % 2),(0, curr_width % 2)], mode='constant', constant_values=BLACK)

    # just a quick check
    assert letter.shape == (final_height, final_width)

    return letter


def normalize_letter_size(letter, width=32, height=32):
    """
        Normalizes the given letter, cropping upper and lower borders 
        and then padding it into the given size (defaults to 32x32)
    """
    letter = cut_upper_and_lower(letter)

    letter = pad(letter, width, height)

    return letter


def crop_image(image, trail):
    """
        Given an image and a snake-trail that succeded in reaching the bottom end of the image, it 
        retrieves the left part of the image as a letter and the right part as the rest of the image
        and returns them
    """

    height, letter_width = 40, max_x(trail)

    rest_width = min_x(trail)

    img_width = image.shape[1]
    
    letter = np.zeros((height, letter_width))
    rest = np.zeros((height, img_width))

    for y, x in trail:
        copy_bits_left(y, x, image, letter)
        copy_bits_right(y, x, image, rest, img_width - 1)
    
    return letter, rest


# ==================== SNAKE FUNCTIONS ==================== #


def surpassing_trail(trail, y, x):
    """
        Checks if the given (y,x) position was already taken by the snake. 
        This avoids looping in the x axis
    """
    return (y,x) in trail


def get_move_set(y, x, max_y, max_x, matrix):
    """
        Computes the possible moves the snake function CAN take and returns them as a tuple 
    """

    up = y > 0 and matrix[y-1][x] != WHITE
    down = y < max_y and matrix[y+1][x] != WHITE
    left = x > 0 and matrix[y][x-1] != WHITE
    right = x < max_x and matrix[y][x+1] != WHITE

    return up, down, left, right


def do_snake(image):
    """
        Finds a set of movements that reach the bottom of the image avoiding white spaces (letters),
        effectively dividing letters from the rest of the image.
        It then crops them as described in the crop_image(...) function and returns the result.
    """
    
    height, width = image.shape
    curr_height = 0
    starting_width = 0

    curr_width = starting_width

    if DEBUG:
        get_ascii_image(image, width)


    debug_image = image.copy()

    snake_moves = []

    while curr_height != height-1:

        snake_moves.append((curr_height, curr_width))

        up, down, left, right = get_move_set(curr_height, curr_width, height-1, width-1, image)

        debug_image[curr_height][curr_width] = 100

        if down:
            curr_height += 1
        elif left and up and not right and not down and not surpassing_trail(snake_moves, curr_height, curr_width-1): # only left and up, move left
            curr_width -= 1
        elif right and up and not left and not down: # only right and up, move right
            curr_width += 1
        # avoid left-right (and viceversa) looping using the surpassing_trail(...) function
        elif right and not surpassing_trail(snake_moves, curr_height, curr_width+1):
            curr_width += 1
        elif left and not surpassing_trail(snake_moves, curr_height, curr_width-1):
            curr_width -= 1
        else: # cannot reach the bottom, restarting from top and delete moves
            snake_moves = []
            starting_width += 1
            curr_height = 0
            curr_width = starting_width

        if DEBUG:
            get_ascii_image(debug_image, width)
            sleep(0.01)

    return crop_image(image, snake_moves)



def snake(index):
    """
        Performs the do_snake function several times, according to the number in the LETTERS_IN_IMAGE var.
        After finding a letter, it normalizes using the normalize_letter_size(...) function and saves it
        to disk for further usage.
    """
    path = f"images/captcha_binary_{index}.png"
    letter_path = "images/captcha_letter_{}_{}.png"

    a = np.array(Image.open(path))
    try:
        for i in range(LETTERS_IN_IMAGE):
            width = a.shape[1]
            a = cut_left_border(a, width - 1)
            letter, a = do_snake(a)
            letter = normalize_letter_size(letter)
            Image.fromarray(letter).convert("L").save(letter_path.format(index, i))
    except Exception as e:
        # print(e)
        # About 1/2 out of 50 samples may fail and throw an exception. 
        # This is a normal behaviour as letters may be overlapping, making it impossible to distinguish between
        # them using the snake movements. 
        # Images that were not cut into letters correctly are prone to failure in the solving process.
        log.warning(f"BAD SNAKE -> {index}")
        pass

def recognize_captcha(captcha):
    try:
        path = "images/captcha_letter_{}_{}.png"

        images = []
        for i in range(LETTERS_IN_IMAGE): # might throw an exception if the images could not be snaked correctly
            images.append(np.array(Image.open(path.format(captcha, i))))

        return predict.predict(images)
    except:
        # it's tricky to deal with the bad snaked images as we need at least 6 images (batch size of trained NN)
        # to predict a result. Right now, we're just avoiding the problem and returning a failure
        return "FAILED" 

def recognize(captcha, results):
    """
        Tries to solve the captcha
    """

    result = recognize_captcha(captcha)

    log.info(f"Result for captcha {captcha} -> {result.encode()}")

    results[f"captcha{captcha}"] = result


def send_results(results, elapsed):
    global base_url, s
    
    log.info("Sending results")

    url = base_url + "/reversecaptcha.php?p=test"

    r = s.post(url, data=results)
    success_percentage = r.text.split("::: ")[1].split(" :::")[0]
    log.info(f"{success_percentage} in {elapsed} seconds")

    if "W3C" in r.text:
        log.success("W3C{" + r.text.split("W3C{")[1].split("}")[0] + "}")
        return True
    return False

def main():

    found_flag = False

    while not found_flag: # repeats if it failed (lesser then 90% of success rate is a failure)
        time_start = time()
        start()
        
        results = {}

        # get images
        for i in range(50):
            get_image(i)

        # break it!
        for i in range(50):
            to_binary(i)
            snake(i)
            recognize(i, results)

        elapsed = time() - time_start
        found_flag = send_results(results, elapsed)


if __name__ == "__main__":
    main()

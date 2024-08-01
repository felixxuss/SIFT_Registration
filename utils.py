import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform


def overlay_images(image1, image2, alpha=0.5):
    """
    Overlay two images with adjustable alpha value.

    Parameters:
        image1 (numpy.ndarray): First input image.
        image2 (numpy.ndarray): Second input image.
        alpha (float): Alpha value for blending. Default is 0.5.

    Returns:
        numpy.ndarray: Overlayed image.
    """
    # Check if images have the same shape
    if image1.shape != image2.shape:
        # Resize the second image to match the shape of the first image
        image2 = transform.resize(image2, image1.shape, preserve_range=True)

    # Blend the images
    blended_image = np.uint((1 - alpha) * image1 + alpha * image2)

    return blended_image


def read_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    return images


def show_image(image, figsize=(6, 6), title=None, cmap="gray"):
    """
    Display an image using matplotlib.

    Parameters:
        image (numpy.ndarray): Input image.
        title (str): Title of the plot. Default is 'Image'.
        cmap (str): Colormap for the plot. Default is 'gray'.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_images(images: list, figsize=(15, 4), title=None):
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    for i, img in enumerate(images):
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
    # supertitle
    plt.suptitle(title)
    plt.show()

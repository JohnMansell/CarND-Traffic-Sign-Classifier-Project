import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import cv2


'''--------------------------
        Show Image
-----------------------------'''
def show(img, img2=None):

    # 1 Image
    if img2 is None:
        plt.imshow(img, cmap='gray')
        plt.show()

    # 2 Images
    if img2 is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))

        axs[0].axis('off')
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Before')

        axs[1].axis('off')
        axs[1].imshow(img2, cmap='gray')
        axs[1].set_title('After')
        plt.show()


'''--------------------------
        Translate
-----------------------------'''
def random_translate(img, shift=None):
    rows = 32
    cols = 32

    # allow translation up to (px) pixels in x and y directions
    if shift is None:
        px = 2
        dx, dy = np.random.randint(-px, px, 2)

    else:
        dx = dy = shift

    # creat matrix M for translation
    M = np.float32([[1,0,dx], [0,1,dy]])

    # Shift image by matrix M
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst


'''--------------------------
        Brightness
-----------------------------'''
def random_brightness(img):

    # Create image of random brightness
    img_max = np.max(img)
    rand_value = np.random.randint((-1 * img_max), img_max)
    brightness = np.copy(img)
    brightness[:] = rand_value

    # Merge Image and Brigness Image
    return cv2.addWeighted(img, 0.5, brightness, 0.5, 0)


'''--------------------------
        Scale
-----------------------------'''
def random_scale(img):
    rows, cols, depth = img.shape

    # transform limits
    px = np.random.randint(-2, 2)

    # Ending Location
    pts1 = np.float32([ [px, px], [rows-px, px], [px, cols-px],  [rows-px, cols-px] ])

    # Starting Locations
    pts2 = np.float32([ [0, 0], [rows, 0], [0, cols], [rows, cols] ])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (rows, cols))
    return dst

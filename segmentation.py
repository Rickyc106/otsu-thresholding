#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class BinarySegmentation(object):
    def segment(self, img, threshold):
        return np.where(img < threshold, 0, 1)

    def otsu_segment(self, img):
        result = np.zeros_like(img)
        highest_variance = 0.
        best_threshold = 0

        for threshold in range(1, 256):
            segmented_img = self.segment(img, threshold=threshold)

            # Compute probabilities of foreground / background segments
            p1 = np.count_nonzero(segmented_img == 0) / np.size(segmented_img)
            p2 = 1. - p1

            # Segmentation is bad if there is only one segment visible
            if p1 == 0 or p2 == 0:
                continue

            # Compute each segment's mean intensity
            mu1 = np.mean(img, where=(segmented_img==0))
            mu2 = np.mean(img, where=(segmented_img==1))

            # Compute between-class variance
            var = p1 * p2 * (mu1 - mu2)**2
            
            # Pick maximum variance
            if var > highest_variance:
                result = segmented_img
                highest_variance = var
                best_threshold = threshold

        return result, best_threshold

def main(args):
    # Read original RGB image
    rgb = Image.open(args.file)

    # Convert image to grayscale
    gray = rgb.convert('L')

    # Convert PIL image to numpy array
    img = np.asarray(gray)

    # Create segmenter object
    segmenter = BinarySegmentation()

    # Perform segmentation
    preset_thresholds = [75, 100, 125]
    s1 = segmenter.segment(img, threshold=preset_thresholds[0])
    s2 = segmenter.segment(img, threshold=preset_thresholds[1])
    s3 = segmenter.segment(img, threshold=preset_thresholds[2])
    otsu, threshold = segmenter.otsu_segment(img)

    # Plot results
    _, axes = plt.subplot_mosaic('ABCC\nDEFG', layout='constrained')
    axes['A'].imshow(rgb)
    axes['A'].set_title('Original Image')
    axes['B'].imshow(gray, cmap='gray')
    axes['B'].set_title('Grayscale Image')
    axes['C'].hist(x=np.arange(256), bins=256, weights=gray.histogram())
    axes['C'].set_title('Grayscale Histogram (256 bins)')
    axes['D'].imshow(s1, cmap='gray')
    axes['D'].set_title(f'Preset Threshold: {preset_thresholds[0]}')
    axes['E'].imshow(s2, cmap='gray')
    axes['E'].set_title(f'Preset Threshold: {preset_thresholds[1]}')
    axes['F'].imshow(s3, cmap='gray')
    axes['F'].set_title(f'Preset Threshold: {preset_thresholds[2]}')
    axes['G'].imshow(otsu, cmap='gray')
    axes['G'].set_title(f'Otsu Threshold: {threshold}')
    plt.show()

if __name__ == '__main__':
    # Parse input arguments
    argparser = argparse.ArgumentParser(description='Performs binary threshold segmentation.')
    argparser.add_argument(
        '--file',
        default='./images/bird.jpg',
        type=str,
        help='File path to image')
    args = argparser.parse_args()

    # Perform algorithm with input args
    main(args)
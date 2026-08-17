import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# IMAGE STATISTICS
# ============================================================

def get_image_statistics(img):

    height, width = img.shape[:2]

    channels = (
        1
        if len(img.shape) == 2
        else img.shape[2]
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    statistics = {
        "Width": width,
        "Height": height,
        "Channels": channels,
        "Mean Brightness": float(np.mean(gray)),
        "Standard Deviation": float(np.std(gray)),
        "Minimum Pixel": int(np.min(img)),
        "Maximum Pixel": int(np.max(img)),
        "Mean Blue": float(np.mean(img[:, :, 0])),
        "Mean Green": float(np.mean(img[:, :, 1])),
        "Mean Red": float(np.mean(img[:, :, 2]))
    }

    return statistics


# ============================================================
# GRAYSCALE HISTOGRAM
# ============================================================

def grayscale_histogram(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        histogram.ravel()
    )

    ax.set_title(
        "Grayscale Histogram"
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_xlim(
        [0, 256]
    )

    fig.tight_layout()

    return fig


# ============================================================
# RGB HISTOGRAM
# ============================================================

def rgb_histogram(img):

    channels = {
        "Blue": 0,
        "Green": 1,
        "Red": 2
    }

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    for name, channel in channels.items():

        histogram = cv2.calcHist(
            [img],
            [channel],
            None,
            [256],
            [0, 256]
        )

        ax.plot(
            histogram.ravel(),
            label=name
        )

    ax.set_title(
        "RGB Histogram"
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_xlim(
        [0, 256]
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

def compare_histograms(
    original,
    processed
):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    processed_gray = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2GRAY
    )

    original_hist = cv2.calcHist(
        [original_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    processed_hist = cv2.calcHist(
        [processed_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        original_hist.ravel(),
        label="Original"
    )

    ax.plot(
        processed_hist.ravel(),
        label="Processed"
    )

    ax.set_title(
        "Original vs Processed Histogram"
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_xlim(
        [0, 256]
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# CHANNEL IMAGES
# ============================================================

def get_rgb_channels(img):

    blue, green, red = cv2.split(img)

    return blue, green, red


# ============================================================
# HSV CHANNELS
# ============================================================

def get_hsv_channels(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    hue, saturation, value = cv2.split(
        hsv
    )

    return hue, saturation, value


# ============================================================
# EDGE DENSITY
# ============================================================

def calculate_edge_density(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(
        edges
    )

    total_pixels = edges.size

    if total_pixels == 0:
        return 0.0

    return (
        edge_pixels / total_pixels
    ) * 100


# ============================================================
# COLOR MEANS
# ============================================================

def get_color_means(img):

    return {
        "Blue": float(
            np.mean(img[:, :, 0])
        ),
        "Green": float(
            np.mean(img[:, :, 1])
        ),
        "Red": float(
            np.mean(img[:, :, 2])
        )
    }
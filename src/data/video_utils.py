"""
Frame helpers shared by the dataset extraction scripts:
square resizing and scene-change detection metrics.
"""
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes the image so that its shortest side equals target_size,
    then applies a center crop to get a perfect square.
    """
    h, w = frame.shape[:2]

    # Aspect-preserving resize
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2

    return resized[start_y:start_y + target_size, start_x:start_x + target_size]


def mse_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pixel-wise Mean Squared Error. A lower value means the images are more similar."""
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))


def histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Bhattacharyya distance between the color histograms of two BGR images,
    averaged over the 3 channels (0 = identical, 1 = very different).
    Typical scene-change threshold: 0.3-0.4.
    """
    dist = 0
    for channel in range(3):  # B, G, R
        hist1 = cv2.calcHist([img1], [channel], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [channel], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        dist += cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return dist / 3


def scene_change_distance(img1: np.ndarray, img2: np.ndarray,
                          thumb_size: int = 16) -> float:
    """
    Mean absolute difference between two downscaled grayscale images, in [0, 1].
    Insensitive to small motions, sensitive to scene changes.
    Typical scene-change threshold: 0.15-0.25.
    """
    t1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    t2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (thumb_size, thumb_size))
    return float(np.mean(np.abs(t1.astype(np.float32) - t2.astype(np.float32))) / 255.0)


def is_scene_change(img1: np.ndarray, img2: np.ndarray,
                    hist_threshold: float = 0.30,
                    thumb_threshold: float = 0.20) -> bool:
    """True if either the histogram or the thumbnail distance exceeds its threshold."""
    return (histogram_distance(img1, img2) > hist_threshold or
            scene_change_distance(img1, img2) > thumb_threshold)


def save_first_frame(video_path: str, image_path: str) -> str:
    """Saves the first frame of a video as an image (e.g. to reuse the input of a result video)."""
    ok, frame = cv2.VideoCapture(str(video_path)).read()
    if not ok:
        raise FileNotFoundError(f"Cannot read video: {video_path}")
    cv2.imwrite(str(image_path), frame)
    return str(image_path)

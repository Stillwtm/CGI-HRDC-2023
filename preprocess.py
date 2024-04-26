import numpy as np
import cv2


def preprocess_image(image, IsGray: bool = False):
    if IsGray:  # return a gray preprocessed image
        image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
    else:  # return a RGB preprocessed image (looks better)
        RADIUS = 400
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), RADIUS / 30), -4, 128)
        circle = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(circle, (RADIUS, RADIUS), int(RADIUS * 0.9), (1, 1, 1), -1, 8, 0)
        image = image * circle + 128 * (1 - circle)

    return image

import numpy as np


class CropPad:
    def __init__(
        self,
        pad_top=3,
        pad_bottom=4,
        pad_left=0,
        pad_right=0,
        crop_top=0,
        crop_bottom=0,
        crop_left=8,
        crop_right=9,
    ):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right

    def __call__(self, image):

        # pad
        image = np.pad(
            image,
            (
                (self.pad_top, self.pad_bottom),
                (self.pad_left, self.pad_right),
            ),
            mode="constant",
            constant_values=0,
        )

        # crop
        h, w = image.shape

        image = image[
            self.crop_top : h - self.crop_bottom,
            self.crop_left : w - self.crop_right,
        ]

        return image


class GlobalZNormalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, images: np.ndarray):
        self.mu = float(images.mean())
        self.sd = float(images.std()) + 1e-8

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sd is None:
            raise RuntimeError("Normalizer must be fitted before transform.")
        return ((image - self.mu) / self.sd).astype(np.float32)
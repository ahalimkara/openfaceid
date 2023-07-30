from __future__ import annotations

import importlib.resources

import dlib
import numpy as np
import PIL.Image
from _dlib_pybind11 import rectangle

_dlib_face_detector = dlib.get_frontal_face_detector()


def _load_image(image_path: str) -> np.ndarray:
    """Load an image from the specified file path and convert it to an RGB numpy array.

    Args:
        image_path: The path to the image file.

    Returns:
        The image as a numpy array with pixel values.
    """

    image = PIL.Image.open(image_path).convert("RGB")

    return np.array(image)


def _get_largest_face_rectangle(image: np.ndarray) -> rectangle | None:
    """Detect the largest face rectangle in the input image.

    Args:
        image: The input image as a numpy array.

    Returns:
        The largest face rectangle. Returns None if no face is detected.
    """

    face_rectangles = _dlib_face_detector(image, 1)

    if len(face_rectangles) == 1:
        return face_rectangles[0]
    elif len(face_rectangles) > 0:
        return max(face_rectangles, key=lambda rect: rect.area())
    else:
        return None


class FaceDetector:
    def __init__(
        self,
        shape_predictor_model_path: str | None = None,
        face_recognition_model_path: str | None = None,
    ):
        if shape_predictor_model_path is None:
            shape_predictor_model_path = str(
                importlib.resources.files(__package__)
                / "models/shape_predictor_5_face_landmarks.dat"
            )

        if face_recognition_model_path is None:
            face_recognition_model_path = str(
                importlib.resources.files(__package__)
                / "models/dlib_face_recognition_resnet_model_v1.dat"
            )

        self.pose_predictor = dlib.shape_predictor(shape_predictor_model_path)
        self.face_recognition = dlib.face_recognition_model_v1(
            face_recognition_model_path
        )

    def get_embeddings(self, image_path: str) -> np.ndarray | None:
        """Compute the face embeddings for the input image.

        Args:
            image_path: The path to the image file.

        Returns:
            The computed face embeddings as a numpy array.
            Returns None if no face is detected in the image.
        """

        image = _load_image(image_path)
        largest_face_rectangle = _get_largest_face_rectangle(image)

        if largest_face_rectangle is None:
            return None
        else:
            landmarks = self.pose_predictor(image, largest_face_rectangle)
            face_vector = self.face_recognition.compute_face_descriptor(
                image, landmarks, 1
            )

            return np.array(face_vector)

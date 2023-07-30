import os

import pytest
from openfaceid.face_detector import FaceDetector

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
_PHOTO_WITH_FACE = os.path.join(_TEST_DATA_DIR, "face.jpg")
_PHOTO_WITHOUT_FACE = os.path.join(_TEST_DATA_DIR, "white.jpg")


@pytest.fixture
def face_detector() -> FaceDetector:
    return FaceDetector()


def test_get_embeddings_returns_correct_embeddings(face_detector: FaceDetector) -> None:
    """Test if the get_embeddings method returns correct embeddings."""
    embeddings = face_detector.get_embeddings(_PHOTO_WITH_FACE)

    assert embeddings is not None
    assert len(embeddings) == 128


def test_get_embeddings_returns_none_for_no_face(face_detector: FaceDetector) -> None:
    """Test if the get_embeddings method returns None when no face is detected."""
    embeddings = face_detector.get_embeddings(_PHOTO_WITHOUT_FACE)

    assert embeddings is None

# OpenFaceID - Identity Verification with Facial Recognition

OpenFaceID is an open-source identity verification project that utilizes facial recognition technology to provide a
robust and secure solution for detecting and authenticating known faces. This project aims to deliver a seamless and
reliable user experience for identity verification in various applications.

## Installation

Use the following command to install OpenFaceID:

```bash
pip install openfaceid
```

## Getting Started

To get started with OpenFaceID, you can follow the FAISS example below:

```python
import faiss
from openfaceid import face_detector
from openfaceid.vector_stores.faiss import FAISS

# Threshold distance used to determine if a face is considered a match or not.
# In this example, it's set to 0.25, which is specific to FAISS and the test images used during testing.
# You may need to adjust this threshold based on your dataset and vector store.
# A lower value means stricter matching, while a higher value allows for more leniency in matching.
_DISTANCE_THRESHOLD = 0.25

fd = face_detector.FaceDetector()

# Get all image ids and vector embeddings of existing faces, i.e. loading from a database
all_embeddings = []

# Initialize the FAISS index and add all image embeddings to it
dimension = len(all_embeddings[0][1])
index = faiss.IndexFlatL2(dimension)
faiss_store = FAISS(index)
faiss_store.add_embeddings(all_embeddings)

# Load the image and perform identity verification
# Replace "path-to/new_selfie.jpg" with the actual path to your new selfie image
face_embedding = fd.get_embeddings(image_path="path-to/new_selfie.jpg")
scores, image_ids = faiss_store.search_with_score(face_embedding)
found = scores[0] < _DISTANCE_THRESHOLD

if found:
    print(f"The face has been successfully matched to an existing face with the id of {image_ids[0]}.")
else:
    print("Unknown selfie.")
```

This code can run when initialising the server. For more examples check `./examples` folder.

## Contributing

We welcome contributions to OpenFaceID! If you have ideas, bug reports, or feature requests, please feel free to open an
issue or submit a pull request. For more information, see CONTRIBUTING.md.

## License

OpenFaceID is released under the Apache License 2.0. Please review the license for more details.

## Acknowledgements

We would like to express our gratitude to the open-source community and the developers of the underlying facial
recognition technology that makes OpenFaceID possible.

## Contact

For questions or inquiries, please contact at ahalimkara@gmail.com

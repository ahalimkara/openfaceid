import pathlib

import pytest


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    # Put all cassettes in ./cassettes/{file_name}/{test}.yaml
    module_path = request.node.path or pathlib.Path(request.module.__file__)
    file_name = module_path.stem

    return str(module_path.parent.joinpath("cassettes", file_name))

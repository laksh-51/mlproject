from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Return a clean list of requirement strings.
    """
    with open(file_path) as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

    # Remove the editable‑install flag if present
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Laksh",
    author_email="lakshkhobragade51@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

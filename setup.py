from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pyroml",
    version="0.0.5",
    author="Nathan Maire",
    author_email="nathan.maire@epfl.ch",
    description="Machine Learning tool allowing plug-and-play training for pytorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(where=".", exclude=["tests"]),
    url="https://github.com/peacefulotter/pyroml",
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "wandb",
    ],
)

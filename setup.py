from setuptools import setup, find_packages

setup(
    name="pyroml",
    version="0.0.1",
    author="Nathan Maire",
    description="Machine Learning tool allowing plug-and-play training for pytorch models",
    long_description="Machine Learning tool allowing plug-and-play training for pytorch models",
    license="MIT",
    packages=find_packages("src", exclude=["tests"]),
    install_requires=[
        "torch",
    ],
)

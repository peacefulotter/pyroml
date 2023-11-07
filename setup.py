from setuptools import setup

setup(
    name="pyro",
    version="0.0.1",
    author="Nathan Maire",
    packages=["pyro"],
    description="Machine Learning tool allowing plug-and-play training for pytorch models ",
    license="MIT",
    install_requires=[
        "torch",
    ],
)

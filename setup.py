from os import path
from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().splitlines()

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    name="sesemi",
    version="0.2.0",
    description="Image Classification with Self-Supervised Regularization",
    author="Flyreel AI",
    author_email="ai@flyreel.co",
    packages=find_packages(include=["sesemi*"]),
    entry_points={"console_scripts": ["open_sesemi = sesemi.trainer.cli:open_sesemi"]},
    url="https://github.com/FlyreelAI/sesemi/",
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
)

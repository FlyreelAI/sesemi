from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().splitlines()

setup(
    name="sesemi",
    version="0.1",
    packages=find_packages(include=["sesemi"]),
    entry_points={"console_scripts": ["open_sesemi = sesemi.trainer.cli:open_sesemi"]},
    url="https://github.com/FlyreelAI/sesemi/",
    include_package_data=True,
    install_requires=requirements,
)

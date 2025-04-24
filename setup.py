# setup.py
from setuptools import setup, find_packages

setup(
    name="yarp",  # This is the name you'll use to import or pip install
    version="0.1",
    packages=find_packages(),  # Automatically finds yarp/ and its subpackages
    install_requires=[],  # Add dependencies here if needed
)

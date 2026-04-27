# setup.py
from setuptools import setup, find_packages

setup(
    name="yarp",  # This is the name you'll use to import or pip install
    version="0.3.0",
    packages=find_packages(),  # Automatically finds yarp/ and its subpackages
    install_requires=[],  # Add dependencies here if needed
    entry_points={  # Allows for easy command-line executables
        'console_scripts': [
            'yarp-init=yarp.initialize_yarp:main',
            'yarp-loop=yarp.run_yarp_loop:main',
            'yarp-progress=yarp.progress_yarp:main',
        ],
    }
)

from setuptools import setup, find_packages

setup(
    name="fbm",  # Name of the package
    version="0.1",
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    author="Zacch Lines",
    description="Simulation of fractional Brownian motion using different methods.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
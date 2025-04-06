from setuptools import setup, find_packages

setup(
    name="rough_bergomi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    author="Zacch Lines",
    author_email="your.email@example.com",
    description="A Python implementation of the rough Bergomi stochastic volatility model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rough_bergomi",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

from setuptools import setup, find_packages
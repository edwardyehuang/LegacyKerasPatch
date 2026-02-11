"""Setup configuration for LegacyKerasPatch."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="legacy-keras-patch",
    version="1.0.0",
    author="Edward Huang",
    author_email="",
    description="Keras 3 ops compatibility layer for Keras 2 + TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edwardyehuang/LegacyKerasPatch",
    packages=find_packages() + ["keras-stubs", "keras-stubs.ops"],
    package_data={
        "legacy_keras_patch": ["py.typed", "ops/*.pyi"],
        "keras-stubs": ["__init__.pyi", "METADATA.toml", "ops/*.pyi"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
        ],
    },
)

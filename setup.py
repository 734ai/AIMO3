"""
setup.py - Package Setup Script

Enables installation of AIMO3 package with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="aimo3-solver",
    version="0.1.0",
    author="AIMO3 Development Team",
    description="AI Mathematical Olympiad (AIMO3) Solver - Solving Olympiad-level math problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aimo3-solver",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="mathematics olympiad AI LLM reasoning SymPy Kaggle",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/aimo3-solver/issues",
        "Documentation": "https://github.com/yourusername/aimo3-solver",
        "Source Code": "https://github.com/yourusername/aimo3-solver",
    },
)

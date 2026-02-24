"""Setup configuration for multimodal_rag package."""

from setuptools import setup, find_packages

setup(
    name="multimodal_rag",
    version="0.1.0",
    description="Production-grade Multimodal RAG 2.0 Assistant",
    author="Your Name",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core dependencies are in requirements.txt
        # This keeps setup.py minimal and requirements.txt as source of truth
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)

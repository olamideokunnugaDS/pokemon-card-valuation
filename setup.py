"""
Setup configuration for pokemon-card-valuation package.
Fixed for Windows encoding issues.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read long description with UTF-8 encoding (Windows fix)
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding='utf-8')
except Exception:
    # Fallback if README can't be read
    long_description = "Multimodal valuation engine for collectible Pokemon cards"

setup(
    name="pokemon-card-valuation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multimodal valuation engine for collectible Pokemon cards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pokemon-card-valuation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pokemon-train-vision=scripts.training.train_vision:main",
            "pokemon-train-market=scripts.training.train_market:main",
            "pokemon-train-fusion=scripts.training.train_fusion:main",
            "pokemon-evaluate=scripts.evaluation.run_evaluation:main",
        ],
    },
)

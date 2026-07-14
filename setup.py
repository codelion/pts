from setuptools import find_packages, setup

# The core -- event schema, storage, classification, linking, migration -- is
# pure Python and imports without torch. Everything that touches a model lives
# behind the `model` extra, which is what `pip install pts` installs by default.
CORE = [
    "tqdm>=4.65.0",
]

MODEL = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    # transformers pins huggingface_hub<1.0; leaving this unbounded lets pip
    # install a 1.x that transformers then refuses to import.
    "huggingface_hub>=0.30.0,<1.0",
    "datasets>=2.12.0",
    "accelerate",
    "numpy>=1.20.0",
    "scikit-learn>=1.0.0",
]

# Sentence PTS uses these for alternative-sentence diversity and for the
# verification pass. Both degrade gracefully when absent.
SENTENCE = [
    "sentence-transformers>=2.2.0",
    "math-verify[antlr4_13_2]>=0.1.0",
]

DEV = [
    "pytest>=7.0",
]

setup(
    name="pts",
    version="2.0.0",
    description=(
        "PTS -- a multiscale causal-event search framework for model reasoning: "
        "pivotal reasoning events at latent, token, and sentence scales"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="codelion",
    author_email="codelion@okyasoft.com",
    url="https://github.com/codelion/pts",
    packages=find_packages(exclude=["tests", "tests.*", "visualizer", "research.*"]),
    install_requires=CORE + MODEL,
    extras_require={
        "core": CORE,
        "model": MODEL,
        "sentence": SENTENCE,
        "dev": DEV,
        "all": MODEL + SENTENCE + DEV,
    },
    entry_points={
        "console_scripts": [
            "pts=pts.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)

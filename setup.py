from setuptools import setup, find_packages

setup(
    name="stereo-block-enc",
    version="0.1.0",
    description="Stereographic projection for quantum block encoding",
    author="",
    author_email="vicenley@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "qiskit>=0.45.0",
        "pennylane>=0.33.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
    },
)

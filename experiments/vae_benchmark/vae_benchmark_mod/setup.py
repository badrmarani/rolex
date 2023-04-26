from setuptools import find_packages, setup

setup(
    name="pythae",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "cloudpickle>=2.1.0",
        "imageio",
        "numpy>=1.19",
        "pydantic>=1.8.2",
        "scikit-learn",
        "scipy>=1.7.1",
        "torch>=1.10.1",
        "tqdm",
        "typing_extensions",
        "dataclasses>=0.6",
    ],
    extras_require={':python_version == "3.7.*"': ["pickle5"]},
    python_requires=">=3.7",
)
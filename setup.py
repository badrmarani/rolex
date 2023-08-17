from setuptools import find_packages, setup

setup(
    name="rolex",
    version="1.0.0",
    description="Robust Latent Space Exploration",
    author="Badr-Eddine Marani",
    author_email="badr-eddine.marani@outlook.com",
    url="https://github.com/badrmarani/rolex",
    install_requires=[
        "pytorch-lightning<=1.9.0",
        "pytorch_sphinx_theme",
        "sphinx_copybutton",
        "sphinx_gallery",
        "pandas<=1.5.3",
        "tensorboard",
        "torchvision",
        "sdmetrics",
        "botorch",
        "einops",
        "ctgan",
        "scipy",
        "numpy",
        "pymoo",
        "tqdm",
    ],
    packages=find_packages(),
)

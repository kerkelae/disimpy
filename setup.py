from setuptools import setup

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="disimpy",
    version="0.3.0",
    description="Massively parallel diffusion MR simulator",
    long_description=long_description,
    url="https://github.com/kerkelae/disimpy",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["disimpy", "disimpy.tests"],
    install_requires=[
        "matplotlib>=3.7.1",
        "numba>=0.58.1",
        "numpy>=1.23.5",
        "pytest>=7.4.4",
        "scipy>=1.11.4",
    ],
    include_package_data=True,
    package_data={"": ["license.txt", "tests/*"]},
)

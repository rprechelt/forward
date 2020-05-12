from os import path
from setuptools import setup
from cmake import CMakeExtension, CMakeBuild

# the apricot version
__version__ = "0.1.0"

# get the absolute path of the python subproject
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# the standard setup info
setup(
    name="forward",
    version=__version__,
    description=("Fourier-regularized Wavelet Deconvolution (ForWaRD) in 1D"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rprechelt/forward",
    author="Remy L. Prechelt",
    author_email="prechelt@hawaii.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["deconvolution", "signal processing", "wavelets"],
    packages=[],
    python_requires=">=3.6*, <4",
    install_requires=["numpy"],
    extras_require={
        "test": ["matplotlib", "mypy", "flake8", "black", "pytest", "coverage"],
    },
    scripts=[],
    project_urls={},
    # call into CMake to build our module
    ext_modules=[CMakeExtension("forward")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

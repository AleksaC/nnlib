from setuptools import setup, find_packages

from nnlib import __version__


with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="nnlib",
    version=__version__,
    author="Aleksa Ćuković",
    author_email="aleksacukovic1@gmail.com",
    description="Simple neural net library built for educational purposes",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/AleksaC/nnlib",
    license="MIT",
    python_requires=">=3.4",
    install_requires=["numpy"],
    tests_require=["pytest"],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)

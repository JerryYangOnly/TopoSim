import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name="skyrsim",
        version="0.2.0",
        author="Yuchen Yang",
        author_email="yuchen@pks.mpg.de",
        description="A package for computing bulk topological invariants and visualizing edge states",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/JerryYangOnly/TopoSim",
        packages=setuptools.find_packages(),
        install_requires=[
            "numpy>=1.16",
            "scipy>=1.5",
            "matplotlib"
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ]
)

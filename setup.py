import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

release_tag = "0.8.23"

setuptools.setup(
    name="planetary-system-stacker",
    version=release_tag,
    author="Rolf Hempel",
    author_email="rolf6419@gmx.de",
    description="PlanetarySystemStacker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rolf-Hempel/PlanetarySystemStacker",
    packages=setuptools.find_packages(),
    ext_modules=None,
    install_requires=[
        'numpy == 1.18.4',
        'mkl',
        'matplotlib',
        'psutil',
        'PyQt5',
        'scipy',
        'astropy',
        'scikit-image',
        'opencv-python'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        # "Operating System :: POSIX :: Linux",
        # "Operating System :: MacOS"
    ],
    python_requires='>=3.5, <3.7',
    entry_points={
        "console_scripts": [
        "PlanetarySystemStacker=planetary_system_stacker.planetary_system_stacker:main",
        ]
    },
)

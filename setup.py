from setuptools import setup,find_packages
import os
path = os.path.join(os.path.dirname(__file__),"python")

with open(os.path.join(path, "jrs/__init__.py"), "r", encoding='utf8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = line.split("'")[1]
            break
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="jrs",
    version=version,
    author="Jittor Group",
    author_email="yuxuan.li.17@ucl.ac.uk",
    description="Jittor remote senisng Detection",
    url="http://jittor.com",
    python_requires='>=3.7',
    packages=find_packages(path),
    package_dir={'': "python"},
    install_requires=[
        "shapely",
        "pyyaml",
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
        "jittor",
        "tensorboardX",
        "opencv-python",
        "tqdm",
        "terminaltables",
    ],
)
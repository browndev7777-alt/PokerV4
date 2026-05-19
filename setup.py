# The MIT License (MIT) — see LICENSE

import codecs
import os
import re
from io import open
from os import path

from setuptools import find_packages, setup


def read_requirements(file_path: str):
    with open(file_path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle.readlines()
            if line.strip() and not line.startswith("#")
        ]


requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()

with codecs.open(
    os.path.join(here, "poker44/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="pokverv3",
    version=version_string,
    description="PokverV3 — Poker44 bot detection miner (SN126)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/browndev7777-alt/PokverV3",
    author="PokverV3 contributors",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)

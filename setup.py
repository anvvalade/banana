#!/usr/bin/env python3

from distutils.core import setup

req_file = open("./requirements.txt")

setup(
    name="banana",
    version="1.0",
    description="Implementation of the Bayesian ANAlysis Natively Adaptable (Banana) Framework",
    author="Aurelien Valade",
    author_email="avalade@aip.de",
    # url='https://www.python.org/sigs/distutils-sig/',
    install_requires=list(req_file.read().splitlines()),
    packages=["banana"],
)

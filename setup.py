#!/usr/bin/env python
# encoding: utf-8

try:
    from setuptools import setup
except ImportError:
    from distutils import setup

setup(
    name="pydc1394",
    version="1.0b",
    description="A Pythonic Wrapper around libdc1394 V2",
    author = "Holger Rapp",
    author_email = "sirver@users.sf.net",
    url="https://launchpad.net/pydc1394",

    packages = [ "pydc1394", "pydc1394.ui", "pydc1394.ui.qt", "pydc1394.ui.wx" ],
)


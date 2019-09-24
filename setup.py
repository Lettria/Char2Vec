import sys
import subprocess
PY_VER = sys.version[0]
subprocess.call(["pip{:} install -r requirements.txt".format(PY_VER)], shell=True)

from setuptools import setup

setup(
    name='chars2vec',
    version='1.0.0',
    author='Lettria',
    original_author='Vladimir Chikin',
    packages=['chars2vec'],
    include_package_data=True,
    package_data={'chars2vec': ['trained_model/*']},
    description="Character-level embedding",
    maintainer='Lettria',
    maintainer_email='hello@lettria.com',
    license='Apache License 2.0',
    classifiers=['Programming Language :: Python :: 3']
)

# -*-coding:utf-8 -*-
import os
import setuptools


with open('requirements.txt') as f:
    requirements = f.readlines()
requirements = [r.strip() for r in requirements]


setuptools.setup(
    name='Moore_SDK',
    version='1.0.20.1',
    description='mooreSDK',
    install_requires=requirements,
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.rst'
        )
    ).read(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    author='Xinjun Wu',
    author_email='wxj@molardata.com',
)

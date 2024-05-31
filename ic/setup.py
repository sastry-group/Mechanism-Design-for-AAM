from setuptools import setup, find_packages

setup(name='ic-bluesky',
    version='0.0.1',
    author='GABY',
    install_requires=[

    ],
    packages=find_packages(
        include=[
            'ic.*'
        ]
    ),
)
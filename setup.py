from setuptools import setup, find_packages

setup(
    name='spreco',
    version='0.0.1',
    description='Training priors for MRI image reconstruction',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tf-slim==1.1.0',
        'numpy',
        'pillow',
        'matplotlib',
        'scikit-image',
        'pyyaml',
        'tqdm'
    ],
)
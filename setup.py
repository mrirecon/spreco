from setuptools import setup, find_packages

setup(
    name='spreco',
    version='0.0.1',
    description='Training priors for MRI image reconstruction',
    packages=find_packages(),
    install_requires=[
        'tensorflow-gpu',
        'tf-slim==1.1.0',
        'numpy',
        'pillow==8.2.0',
        'matplotlib',
        'scikit-image',
        'pyyaml==5.4.1',
        'tqdm'
    ],
)
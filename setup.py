from setuptools import setup, find_packages

setup(
    name='spreco',
    version='0.0.1',
    description='Training priors for MRI image reconstruction',
    packages=find_packages(),
    install_requires=[
        'tensorflow-gpu==2.4.1',
        'tf-slim==1.1.0',
        'numpy',
        'pillow==8.2.0',
        'matplotlib==3.3.4',
        'scikit-image==0.18.1',
        'pyyaml==5.4.1',
        'tqdm'
    ],
)
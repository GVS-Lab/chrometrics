from setuptools import setup

setup(
    name='nmco',
    version='0.1',
    packages=['nmco','nmco.nuclear_features','nmco.utils'],
    url='https://github.com/GVS-Lab/NMCO-Image-Features.git',
    license='MIT',
    author='Saradha Venkatachalapathy',
    author_email='saradhavpathy@gmail.com',
    description='Nuclear morphology extraction package for 2D developed by Saradha Venkatachalapathy',
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.1.2',
        'matplotlib>=3.3.2',
	'opencv-python-headless>=4.4.0.42',
	'tifffile>=2020.10.1',
	'scikit-image>=0.17.2',
	'scipy>=1.5.2',
	'scikit-learn>=0.23.2',
	'tqdm>=4.50.0',
    ]
)

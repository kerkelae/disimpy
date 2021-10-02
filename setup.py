from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='disimpy',
    version='0.1.2',
    description='Massively parallel diffusion MR simulator',
    url='https://github.com/kerkelae/disimpy',
    license='MIT',
    packages=['disimpy', 'disimpy.tests'],
    install_requires=requirements,
    include_package_data=True,
    package_data={'': ['tests/*']},
)

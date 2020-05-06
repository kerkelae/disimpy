from distutils.core import setup

setup(
    name='disimpy',
    version='0.1',
    description='Functions for Monte Carlo random walker dMRI simulations.',
    url='https://github.com/kerkelae/disimpy',
    license='MIT',
    packages=['disimpy',],
    install_requires=['numpy','numba','dipy','matplotlib','scipy'],
)

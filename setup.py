from distutils.core import setup

setup(
    name='disimpy',
    version='0.1',
    description='Package for simulating diffusion MRI experiments using massively parallelized monte carlo random walker diffusion simulations.',
    url='https://github.com/kerkelae/disimpy',
    license='MIT',
    packages=['disimpy',],
    install_requires=['numpy','numba','matplotlib','scipy'],
)

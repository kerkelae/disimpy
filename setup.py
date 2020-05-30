from distutils.core import setup

setup(
    name='disimpy',
    version='0.1',
    description='Package for simulating diffusion MRI experiments on CUDA GPU.',
    url='https://github.com/kerkelae/disimpy',
    license='MIT',
    packages=['disimpy',],
    install_requires=[
        'numpy>=1.18.4',
        'matplotlib>=3.1.3',
        'numba>=0.49.0',
        'scipy>=1.4.1',
        'pytest>=5.4.1',
    ],
)

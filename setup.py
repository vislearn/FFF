from setuptools import setup, find_packages

setup(
    name='fff',
    description='Free-form flows',
    version='0.1dev',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "torch",
        "numpy"
    ],
    license='MIT',
    author='Felix Draxler, Peter Sorrenson',
    author_email='felix.draxler@iwr.uni-heidelberg.de, peter.sorrenson@iwr.uni-heidelberg.de',
    url='https://github.com/vislearn/FFF'
)

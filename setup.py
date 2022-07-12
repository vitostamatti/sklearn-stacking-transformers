from setuptools import setup, find_packages


setup(
    name='stacking_transformer',
    version='0.1.0',
    description='transformer for stacking of models',
    # long_description=long_description,
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
    install_requires=[
        'scikit-learn', 
    ],
),
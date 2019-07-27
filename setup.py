import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydistinct",
    version="0.3",
    author="Edwin Chan",
    author_email="edwinchan@u.nus.edu",
    description="Package for estimating distinct values in a population",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chanedwin/pydistinct/",
    download_url='https://github.com/chanedwin/pydistinct/archive/v0.3.tar.gz', 
    keywords=['distinct', 'value', 'estimators', 'sample', 'sequences'], 
    install_requires=[ 
        'scipy',
        'statsmodels',
        'xgboost',
        'numpy'
    ],

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

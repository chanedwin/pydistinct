import setuptools

setuptools.setup(
    name="pydistinct",
    version="0.6.2",
    author="Edwin Chan",
    author_email="edwinchan@u.nus.edu",
    description="Package for estimating distinct values in a population",
    long_description=""" This package provides statistical estimators to predict a population's total number of distinct values from a sample sequence - given a sample sequence of n values with only d distinct values, predict the total number of distinct values D that exists in the population N.

    Sample use cases :
    
    estimating the number of unique insects in a population from a field sample
    estimating the number of unique words in a document given a sentence or a paragraph
    estimating the number of unique items in a database from a few sample rows
    Please send all bugs reports/issues/queries to chanedwin91@gmail.com for fastest response!
    
    See https://github.com/chanedwin/pydistinct for more information 
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/chanedwin/pydistinct/",
    download_url='https://github.com/chanedwin/pydistinct/archive/0.5.tar.gz',
    keywords=['distinct', 'value', 'estimators', 'sample', 'sequences'],
    install_requires=[
        'scipy',
        'statsmodels',
        'xgboost',
        'numpy',
        'm2r'
    ],

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

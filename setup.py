from setuptools import setup, find_packages

setup(
    name='lda_eval_lib',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'openai==1.59.8',
        'pyLDAvis==3.4.1',
        'scikit-learn==1.5.1',
        'scipy==1.10.1'
     ],
    python_requires='>=3.9, <3.10',  # Specify Python version compatibility if necessary
    include_package_data=True,
    package_data={
        "": ["test/*.pkl"]  # Include all .pkl files in the 'test' folder
    }
)

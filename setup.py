from setuptools import setup, find_packages

setup(
    name='lda_eval_lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gensim==3.8.3',
        'numpy==2.2.2',
        'openai==1.59.8',
        'pandas==2.2.3',
        'pyLDAvis==3.4.1',
        'scikit_learn==1.6.1',
        'setuptools==69.5.1'
     ],
    python_requires='>=3.9',  # Specify Python version compatibility if necessary
)
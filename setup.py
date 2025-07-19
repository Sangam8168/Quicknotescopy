from setuptools import setup, find_packages

setup(
    name="quicknotes",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'requests',
        'torch',
        'transformers',
        'nltk',
        'youtube-transcript-api',
        'PyPDF2',
        'gunicorn',
    ],
    python_requires='>=3.9',
)

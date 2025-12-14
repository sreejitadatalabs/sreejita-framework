from setuptools import setup, find_packages

setup(
    name="sreejita-framework",
    version="1.7.0",    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "reportlab>=4.0",
        "python-dateutil>=2.8",
        "pyaml>=23.7",
        "watchdog>=3.0",
        "psutil>=5.9"
    ],
    python_requires=">=3.8",
    author="Yeswanth Arasavalli",
    description="Universal Data Analytics & Reporting Engine with Quality Assurance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sreejitadatalabs/sreejita-framework",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]
)

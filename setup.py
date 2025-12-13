from setuptools import setup, find_packages

setup(
    name="sreejita-framework",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas","numpy","pyyaml","scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "sreejita=sreejita.cli:main"
        ]
    }
)

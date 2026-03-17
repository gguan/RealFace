from setuptools import setup, find_packages

setup(
    name="faceforge",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": ["faceforge=faceforge.cli:app"],
    },
)

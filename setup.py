from setuptools import setup, find_packages

setup(
    name="tem_agent",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "imageio>=2.9.0",
        "tqdm>=4.62.0",
    ],
    description="TEM image analysis tools for FinFET structures using LLM agents",
    author="Your Name",
) 
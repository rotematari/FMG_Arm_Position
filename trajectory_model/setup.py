from setuptools import setup, find_packages

setup(
    name="trajectory_model",
    version="0.1.0",
    description="A package for forecasting time series data from location-based time series",
    author="Rotem Atari",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy",         
        "pandas",
        "torch",    
        "matplotlib",
        "scikit-learn",
        "scipy",
        
    ],
    python_requires='>=3.6',
)
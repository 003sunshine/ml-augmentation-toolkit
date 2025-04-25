from setuptools import setup, find_packages

setup(
    name='alloyxai',
    version='0.1.0',
    description="An integrated machine learning pipeline for advanced data augmentation and model interpretability in high-temperature alloy research.",
    author_email='linlinsun1010@163.com',
    url='https://github.com/003sunshine/alloyxai',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'shap>=0.41.0',
        'smogn>=0.1.2',
        'pymc>=5.0.0',
        'arviz>=0.12.0',
        'torch>=1.9.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)

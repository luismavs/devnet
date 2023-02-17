from setuptools import find_namespace_packages, setup, find_packages

setup(
    name="devnet",
    version="0.1.1",
    description="Unofficial pytorch implementation of deviation network for table data.",
    author="Yuji Kamiya",
    author_email="y.kamiya0@gmail.com",
    license="MIT",
    packages=find_packages(where='src'),
    # packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires="~=3.8",
    install_requires=["pandas", "torch", 'scikit-learn'],
    zip_safe=False,
)

# from Cython.Build import cythonize
# ext_modules=cythonize("anomalydetector/inwkitsune/feature_extractor/after_image_cython.pyx"),

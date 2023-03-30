from setuptools import setup, find_packages

setup(
    name="devnet",
    version="0.3.2",
    description="Unofficial pytorch implementation of deviation network for table data.",
    author="Yuji Kamiya",
    author_email="y.kamiya0@gmail.com",
    maintainer="Luís Seabra",
    maintainer_email="luismavseabra@gmail.com",
    license="MIT",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=["pandas", "torch", 'scikit-learn', 'logzero','hydra-core'],
    zip_safe=False,
)

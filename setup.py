from setuptools import setup, find_packages

setup(
    name="inw-devnet",
    version="0.4.1",
    description="Unofficial pytorch implementation of deviation network for table data.",
    author="Yuji Kamiya",
    author_email="y.kamiya0@gmail.com",
    maintainer="Luís Seabra",
    maintainer_email="luismavseabra@gmail.com",
    license="MIT",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=["pandas", "torch", 'scikit-learn', 'logzero'],
    extras_require={"hydra": 'hydra-core'},
    zip_safe=False,
)

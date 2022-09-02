from setuptools import setup, find_packages

setup(
    name="pyutil",
    version="0.0.1",
    description="Python Utilites",
    author="Panda Pan",
    author_email="panchongdan@gmail.com",
    packages=find_packages(),
    install_requires=["pandas>=1.4.0"],
    tests_require=["pytest>=3.3.1", "pytest-cov>=2.5.1"],
    python_requires=">=3.8",
)

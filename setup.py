import setuptools

setuptools.setup(
    name="semantify",
    version="0.0.1",
    author="Omer Gralnik",
    author_email="omergral@gmail.com",
    description="Semantify repository",
    package_data={"": ["license.txt"]},
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


with open("README.md") as readme_file:
    readme = readme_file.read()

print(readme)

requirements = read_requirements(r"requirements.txt")
# docs_extras = read_requirements(r'requirements_dev.txt'))
test_requirements = [
    "pytest>=3",
]
# fmt: off
setup(
    author="Mauro Luzzatto",
    author_email="mauroluzzatto@hotmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description=(
        "explainy is a library for generating explanations for machine learning models"
        " in Python. It uses methods from Machine Learning Explainability and provides"
        " a standardized API to create feature importance explanations for samples. The"
        " explanations are generated in the form of plots and text."
    ),
    entry_points={
        "console_scripts": [
            "explainy=explainy.cli:main",
        ],
    },
    install_requires=requirements,
    # extras_require={'docs': docs_extras},
    license="MIT license",
    # long_description=f'"""{readme}"""',
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="explainy",
    name="explainy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MauroLuzzatto/explainy",
    version='0.2.9',
    zip_safe=False,
)
# fmt: on

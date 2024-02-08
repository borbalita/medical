from typing import List

from setuptools import find_packages, setup


def _get_requirements(env: str = "") -> List[str]:
    requirements_file = f"requirements{'-' + env if env else ''}.txt"
    with open(requirements_file, "r") as file:
        return file.read().splitlines()


setup(
    name='pneumonia',
    version='0.1.0',
    url='https://github.com/borbalita/public/medical/pneumonia',
    author='Borbala Tasnadi',
    author_email='borbala.tasnadi@gmail.com',
    description='App for Pneumonia detection in X-ray images',
    packages=find_packages(),
    install_requires=_get_requirements(),
    extras_require={
        'dev': _get_requirements('dev')
    },
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

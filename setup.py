from setuptools import find_packages, setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='pnm',
    version='0.1.0',
    url='https://github.com/borbalita/public/medical/pneumonia',
    author='Borbala Tasnadi',
    author_email='borbala.tasnadi@gmail.com',
    description='App for Pneumonia detection in X-ray images',
    packages=find_packages(),
    install_requires=required,
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

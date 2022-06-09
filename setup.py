from setuptools import setup
from setuptools import find_packages


with open("docs/README.PyPI.md", "r") as fh:
    long_description = fh.read()

setup(
    name='aucmedi',
    version='0.6.0',
    description='AUCMEDI - a framework for Automated Classification of Medical Images',
    author='Dominik Müller',
    author_email='dominik.mueller@informatik.uni-augsburg.de',
    license='GPLv3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://frankkramer-lab.github.io/aucmedi/',
    project_urls={
        "Bug Tracker": "https://github.com/frankkramer-lab/aucmedi/issues",
        "Documentation": "https://frankkramer-lab.github.io/aucmedi/reference/",
        "Source Code": "https://github.com/frankkramer-lab/aucmedi",
    },
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['tensorflow>=2.6.0',
                      'keras-applications>=1.0.8',
                      'numpy>=1.19.2',
                      'pillow>=8.3.2',
                      'albumentations>=1.1.0',
                      'pandas>=1.4.0',
                      'scikit-learn>=1.0.2',
                      'scikit-image>=0.19.1',
                      'lime>=0.2.0.1',
                      'pooch>=1.6.0',
                      'classification-models-3D>=1.0.4',
                      'tensorflow-addons>=0.16.1',
                      'vit-keras>=0.1.0',
                      'SimpleITK>=2.1.1',
                      'batchgenerators>=0.23',
                      'volumentations-aucmedi>=1.0.1',
                      'plotnine>=0.8.0'],
    classifiers=["Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.10",
                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Operating System :: OS Independent",

                 "Intended Audience :: Healthcare Industry",
                 "Intended Audience :: Science/Research",

                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Topic :: Scientific/Engineering :: Image Recognition",
                 "Topic :: Scientific/Engineering :: Medical Science Apps."]
    )

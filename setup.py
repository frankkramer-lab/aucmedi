from setuptools import setup
from setuptools import find_packages


with open("docs/README.PyPI.md", "r") as fh:
    long_description = fh.read()

setup(
    name='aucmedi',
    version='0.11.0',
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
    entry_points={
        'console_scripts': ['aucmedi = aucmedi.automl.main:main'],
    },
    python_requires='>=3.9',
    install_requires=['tensorflow>=2.14.0',
                      'numpy>=1.23.5',
                      'pillow>=10.2.0',
                      'albumentations>=1.3.0',
                      'pandas>=1.5.2',
                      'scikit-learn>=1.3.0',
                      'scikit-image>=0.21.0',
                      'lime>=0.2.0.1',
                      'pooch>=1.6.0',
                      'classification-models-3D>=1.0.10',
                      # 'vit-keras>=0.1.2',
                      'Keras-Applications==1.0.8',
                      'SimpleITK>=2.2.0',
                      'batchgenerators>=0.25',
                      'volumentations-aucmedi>=1.0.1',
                      'plotnine==0.12.4',
                      'pathos>=0.3.0'],
    classifiers=["Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.10",
                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Operating System :: OS Independent",

                 "Intended Audience :: Healthcare Industry",
                 "Intended Audience :: Science/Research",

                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 "Topic :: Scientific/Engineering :: Image Recognition",
                 "Topic :: Scientific/Engineering :: Medical Science Apps."]
    )

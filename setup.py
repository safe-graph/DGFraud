from setuptools import setup
from setuptools import find_packages

setup(name='DGFraud',
      version='1.0',
      description='a GNN based toolbox for fraud detection in Tensorflow',
      download_url='https://github.com/safe-graph/DGFraud',
      install_requires=['numpy>=1.16.4',
                        'tensorflow>=1.14.0,<2.0',
                        'scipy>=1.2.0'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())
from setuptools import find_packages, setup
from typing import List

REQUIREMENTS_FILE_NAME = 'requirements.txt'

HYPHEN_E_DOT = "-e ."

def get_requirements_list()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as f:
        requirement_list = f.readlines()
        requirement_list = [i.replace("\n","") for i in requirement_list]

        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)

        return requirement_list


setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This app will help all music enthusiasts to predict success/failure of a new music, get new song recommendations, insights module and analytics all in one place',
    author='SrijanDeo',
    license='',
    install_requires = get_requirements_list()
)

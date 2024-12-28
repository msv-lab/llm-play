from setuptools import setup, find_packages

setup(
    name="llm-query",
    version="1.0.0",
    packages=find_packages(),
    py_modules=["llm_query"],
    install_requires=[
        'pyyaml',
        'anthropic',
        'openai',
        'inquirer',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'llm-query=llm_query:main',
        ],
    },
)

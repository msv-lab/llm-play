from setuptools import setup, find_packages

setup(
    name="llm-play",
    version="1.0.0",
    packages=find_packages(),
    py_modules=["llm_play"],
    install_requires=[
        'pyyaml',
        'anthropic',
        'openai',
        'InquirerPy',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'llm-play=llm_play:main',
        ],
    },
)

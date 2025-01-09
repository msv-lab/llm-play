from setuptools import setup, find_packages

setup(
    name="llm-play",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["llm_play"],
    install_requires=[
        'pyyaml',
        'anthropic',
        'openai',
        'InquirerPy',
        'wcwidth',
        'mistletoe'
    ],
    entry_points={
        'console_scripts': [
            'llm-play=llm_play:main',
        ],
    },
)

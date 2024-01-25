from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name="koasr",
    packages=["koasr"],
    version="0.0.1",
    license="MIT",
    description="An open-source framework for fine-tunning whisper model",
    install_requires=required
)
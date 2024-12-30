from setuptools import setup, find_packages

setup(
    name="ah_runpod_sd",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "runpod",
        "nanoid",
        "asyncio",
        "aiohttp",
        "Pillow"
    ],
)

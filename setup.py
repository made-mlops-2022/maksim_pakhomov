from setuptools import find_packages, setup


setup(
    name="ml",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Maksim Pakhomov",
    entry_points={
        "console_scripts": [
            "ml_train = ml.train_pipeline:train_pipeline_func"
        ]
    },
    #install_requires=required,
    license="MIT",
)

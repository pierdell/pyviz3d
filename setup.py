import setuptools

setuptools.setup(
    name="pdell",  # Replace with your username

    version="1.0.0",

    author="pierre dellenbach",

    author_email="<pierre.dellenbach@kitware.com>",

    description="viz3d is an interactive toy visualizer for 3D Data with numpy",

    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)

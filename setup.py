from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='VolumetricSMPL',
    version='1.0.4+jax',
    packages=['VolumetricSMPL'],
    url='https://github.com/markomih/VolumetricSMPL',
    license='MIT',
    author='Marko Mihajlovic',
    author_email='markomih@inf.ethz.ch',
    description='VolumetricSMPL body model, unofficial JAX port.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This tells PyPI it's Markdown
    python_requires='>=3.11',
    install_requires=[
        'jax',
        'numpy',
        'trimesh',
        'scikit-image',
    ],
)

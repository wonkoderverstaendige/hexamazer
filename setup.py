try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='hexamazer',
      description='HexAMazer',
      author='Ronny Eichler',
      author_email='ronny.eichler@gmail.com',
      version='0.0.4',
      install_requires=['numpy', 'pandas'],
      packages=['hexamazer'],
      entry_points="""[console_scripts]
            hexamazer=hexamazer.main:main""")
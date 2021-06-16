from setuptools import setup

setup( name = 'rainier',
       author = 'Niket Thakkar',
       author_email = 'nthakkar@idmod.org', 
       description = ('library associated with the report: https://covid.idmod.org/data/One_state_many_outbreaks.pdf'),
       packages = ['rainier'],
       install_requires = [ 'matplotlib', 
                            'numpy',
                            'pandas==1.0.5',
                            'scipy',
                            'tqdm'
                           ]
      )

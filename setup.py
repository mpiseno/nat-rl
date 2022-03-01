from setuptools import setup


setup(name='nat_rl',
      version='0.1',
      description='Training tools for natural-language RL',
      packages=['nat_rl'],
      install_requires=[
            'numpy', 'gym', 'matplotlib', 'scipy',
            'stable-baselines3',
      ])

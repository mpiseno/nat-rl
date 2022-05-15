from setuptools import setup


setup(name='nat_rl',
      version='0.1',
      description='Training tools for natural-language RL',
      packages=['nat_rl'],
      install_requires=[
            'numpy', 'gym', 'matplotlib', 'scipy',
            'stable-baselines3',

            'pybullet==3.0.4',
            'pygame==2.0.1',

            'imitation @ git+https://github.com/HumanCompatibleAI/imitation.git#sha1=ed45793dfdd897d3ac1f3a863a8816b56d436887',

            # CLIP reqs
            'clip @ git+https://github.com/openai/CLIP.git',
            'ftfy', 'regex', 'tqdm'
      ])

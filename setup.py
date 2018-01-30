from setuptools import setup

if __name__ == '__main__':
    readme = open('README.md').read()
    import pypandoc
    long_description = pypandoc.convert_text(
        readme, 'rst', format='markdown_github')

    setup(long_description=long_description,
          entry_points = {
              'console_scripts': [
                  'runAssociation=limmbo.bin.runAssociation:entry_point',
                  'runVarianceEstimation=limmbo.bin.runVarianceEstimation:entry_point'
                  ]
              }
          )

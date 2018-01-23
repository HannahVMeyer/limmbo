from setuptools import setup

if __name__ == '__main__':
    readme = open('README.md').read()
    try:
        import pypandoc
        long_description = pypandoc.convert_text(
            readme, 'rst', format='markdown_github')
    except (ImportError, RuntimeError, OSError):
        long_description = readme

    setup(long_description=long_description,
          entry_points = {
              'console_scripts': ['runGWAS=limmbo.bin.runGWAS:entry_point'],
              'console_scripts': ['runVarianceEstimation=limmbo.bin.runVarianceEstimation:entry_point']
              }
          )

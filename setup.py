from setuptools import find_packages, setup


def parse_requirements(f):
    with open(f, 'r') as fp:
        for line in fp.readlines():
            yield line.strip()


def main():
    setup(name='waveglow',
          version='0.0.1',
          author='なるみ',
          author_email='weaper@gamil.com',
          packages=find_packages(),
          install_requires=list(parse_requirements('requirements.txt')))


if __name__ == "__main__":
    main()

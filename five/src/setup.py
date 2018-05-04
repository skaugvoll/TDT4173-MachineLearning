

def try_or_install(package):
    try:
        import package
    except Exception as e:
        import pip

        pip.main(['install', package])



def main():
    packages = [
        "scipy",
        "numpy",
        "pillow",
        "tensorflow"
    ]
    try_or_install(packages)

if __name__ == "__main__":
    main()
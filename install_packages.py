import subprocess

def install_packages(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            package = line.strip()
            if package and not package.startswith('#'):
                try:
                    subprocess.run(['python3', '-m', 'pip' ,'install', package], check=True)
                    print(f"Successfully installed {package}")
                except subprocess.CalledProcessError:
                    print(f"Failed to install {package}, skipping...")

if __name__ == "__main__":
    install_packages('requirements.txt')
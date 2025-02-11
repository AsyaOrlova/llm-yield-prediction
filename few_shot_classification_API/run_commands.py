import os
import subprocess
from tqdm import tqdm

config_directory = "your config directory"
command = "python -m classifier"
config_files = os.listdir(config_directory)
progress_bar = tqdm(total=len(config_files), unit="file")

paths = []
for config_file in config_files:
    config_path = os.path.join(config_directory, config_file)
    if os.path.isfile(config_path) and config_file.endswith(".txt"):
        paths.append(config_path)
        print(config_path)
        full_command = f"{command} {config_path}"

        try:
            subprocess.run(full_command, shell=True, check=True)
            progress_bar.set_description(f"Run success: {config_file}")
        except subprocess.CalledProcessError as e:
            progress_bar.set_description(f"Run error: {config_file}")
            print(f"Error: {e.returncode}")

    progress_bar.update(1)

progress_bar.close()
print(paths)
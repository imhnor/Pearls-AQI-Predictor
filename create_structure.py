import os

structure = [
    "data/.gitkeep",
    "models/.gitkeep",

    "src/fetch_data.py",
    "src/features.py",
    "src/train.py",
    "src/predict.py",

    "app.py",
    "requirements.txt",
    "README.md"
]

project_name = "aqi-project"

for path in structure:
    full_path = os.path.join(project_name, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w") as f:
        pass  # create empty file

print("AQI project structure created successfully.")
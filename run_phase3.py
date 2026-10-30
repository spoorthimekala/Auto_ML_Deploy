
from pathlib import Path

project_dir = Path(__file__).parent

required_folders = [
    "automl",
    "models",
    "uploads",
    "templates",
    "static",
    "sample_data",
]

print("Checking project folders...")

for folder in required_folders:
    path = project_dir / folder
    print(f"{folder}: {'OK' if path.is_dir() else 'MISSING'}")

print("\nFolder check completed.")

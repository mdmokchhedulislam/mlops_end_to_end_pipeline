import os
from pathlib import Path

# প্রজেক্টের মূল ফোল্ডারের নাম
PROJECT_NAME = "fraud-detection-mlops"

# ফোল্ডার এবং ফাইলসমূহের তালিকা
STRUCTURE = [
    # Directories
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/training",
    "src/evaluation",
    "src/serving",
    "tests",
    "configs",
    "workflows",
    "docker",
    "k8s",
    "monitoring",
    "notebooks",
    # Root Files
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "Dockerfile",
    "Makefile",
    "README.md",
]


def create_project_structure(root_dir, structure):
    base_path = Path(root_dir)

    for item in structure:
        target_path = base_path / item

        # যদি এক্সটেনশন বা নির্দিষ্ট ফাইল হয় (যেমন .yaml, .txt, .md, Dockerfile, Makefile)
        if "." in target_path.name or target_path.name in [
            "Dockerfile",
            "Makefile",
        ]:
            # ফাইল তৈরির আগে তার প্যারেন্ট ডিরেক্টরি তৈরি করে নিবে
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                target_path.touch()
                print(f"File created: {target_path}")
            else:
                print(f"File already exists: {target_path}")
        else:
            # এটি একটি ডিরেক্টরি
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {target_path}")


if __name__ == "__main__":
    create_project_structure(PROJECT_NAME, STRUCTURE)
    print("\nProject structure creation completed successfully!")
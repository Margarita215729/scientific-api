import os
import json
from drive.drive_client import get_universe_tree

def ensure_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")

def generate_snapshot():
    print("📦 Генерация snapshot дерева папки Universe...")
    tree = get_universe_tree(max_depth=10)
    ensure_data_folder()
    with open("data/universe_tree_snapshot.json", "w") as f:
        json.dump(tree, f, indent=2)
    print("✅ Snapshot сохранён: data/universe_tree_snapshot.json")

if __name__ == "__main__":
    generate_snapshot()

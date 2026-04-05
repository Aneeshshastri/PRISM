import json
import io

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"
with io.open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"# --- CELL {i} ---")
        print("".join(cell.get('source', [])))
        print("\n" + "="*40 + "\n")

import json

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if 'def run_inference_pipeline' in source_text:
        print(f"Cell {i} contains run_inference_pipeline")
        print("-" * 20)
        print(source_text)
        print("=" * 40)
    if 'report_map_results(' in source_text:
        print(f"Cell {i} calls report_map_results")
        print("-" * 20)
        print(source_text)
        print("=" * 40)

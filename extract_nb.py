import json
import io

nb_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"
txt_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\notebook_source.txt"

def extract():
    with io.open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code.append("".join(cell.get('source', [])))
    
    with io.open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# " + "="*60 + "\n# NEW CELL\n# " + "="*60 + "\n\n")
        f.write("\n\n# " + "="*60 + "\n# NEW CELL\n# " + "="*60 + "\n\n".join(code))

extract()
print(f"Extracted code to {txt_path}")

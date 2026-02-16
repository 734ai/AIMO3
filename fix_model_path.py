import json
import os

NOTEBOOK_PATH = "notebooks/aimo3_kaggle_ready.ipynb"

def fix_model_path():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    cells = nb["cells"]
    
    # Target the cell with AIOMInference.load_model
    target_content = "potential_paths = ["
    
    fixed = False
    for cell in cells:
        if cell["cell_type"] == "code":
            source_str = "".join(cell["source"])
            if target_content in source_str and "AIOMInference" in source_str:
                print("Found load_model cell. Updating...")
                
                new_source = []
                for line in cell["source"]:
                    new_source.append(line)
                    if "f\"/kaggle/input/{self.model_name.replace('/', '-')}\"" in line:
                        new_source.append(",\n")
                        new_source.append("            \"/kaggle/input/model-gpt2\", # Custom dataset path\n")
                        new_source.append("            \"/kaggle/input/muzansano/model-gpt2\" # Full path variant\n")
                
                cell["source"] = new_source
                fixed = True
                break
    
    if fixed:
        with open(NOTEBOOK_PATH, 'w') as f:
            json.dump(nb, f, indent=1)
        print("✅ Fixed notebook model paths.")
    else:
        print("❌ Could not find load_model cell to fix.")

if __name__ == "__main__":
    fix_model_path()

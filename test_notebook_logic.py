import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.abspath("src"))

NOTEBOOK_PATH = "notebooks/aimo3_kaggle_ready.ipynb"

def run_notebook_logic():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    
    # Mock heavy dependencies for logic testing
    import sys
    from unittest.mock import MagicMock
    
    sys.modules["torch"] = MagicMock()
    sys.modules["torch"].__version__ = "2.0.0"
    sys.modules["transformers"] = MagicMock()
    sys.modules["transformers"].__version__ = "4.30.0"
    sys.modules["peft"] = MagicMock()
    sys.modules["accelerate"] = MagicMock()
    
    print("✅ Mocked torch, transformers, peft, accelerate for logic testing")
    
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]
    
    print(f"Found {len(code_cells)} code cells.")
    
    # Context for execution
    context = {}
    
    for i, source in enumerate(code_cells):
        print(f"\n--- Executing Cell {i+1} ---")
        code = "".join(source)
        
        # Skip pip install
        if "!pip install" in code:
            print("Skipping pip install cell")
            continue
            
        try:
            exec(code, context)
            print("✅ Cell executed successfully")
        except Exception as e:
            print(f"❌ Error in Cell {i+1}: {e}")
            # Don't exit, try to continue to see if subsequent cells fail
            # But usually a failure here stops the notebook flow.
            # We break on error for safety in this test
            break

if __name__ == "__main__":
    run_notebook_logic()

import json
import os

NOTEBOOK_PATH = "notebooks/aimo3_kaggle_ready.ipynb"

def fix_imports():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    cells = nb["cells"]
    
    # Target the cell with sys.path logic I added previously
    # It ends with: sys.path.insert(0, os.getcwd())
    
    target_content = "sys.path.insert(0, os.getcwd())"
    
    new_source = [
        "# Add phase 4 source code to path\n",
        "# Try multiple potential paths for robustness\n",
        "import sys\n",
        "import os\n",
        "\n",
        "potential_src_paths = [\n",
        "    '/kaggle/input/aimo-solver-phase4',      # Flat upload\n",
        "    '/kaggle/input/aimo-solver-phase4/src',  # Folder upload\n",
        "    'src',                                   # Local\n",
        "    '../src'                                 # Local relative\n",
        "]\n",
        "\n",
        "src_path_added = False\n",
        "for path in potential_src_paths:\n",
        "    if os.path.exists(path):\n",
        "        if path not in sys.path:\n",
        "            sys.path.insert(0, path)\n",
        "            print(f\"✅ Added to sys.path: {path}\")\n",
        "        src_path_added = True\n",
        "        break\n",
        "\n",
        "if not src_path_added:\n",
        "    print(\"⚠️ Could not find source code path!\")\n",
        "    if os.getcwd() not in sys.path:\n",
        "        sys.path.insert(0, os.getcwd())\n",
        "\n",
        "# Phase 4: Import Verification & Metrics Components\n",
        "try:\n",
        "    from monitoring import VerificationTracker, ExecutionMetrics\n",
        "    from resilience import ErrorRecoveryHandler\n",
        "    from computation import SymbolicCompute, AnswerValidator\n",
        "    \n",
        "    PHASE4_AVAILABLE = True\n",
        "    print(\"✅ Phase 4 components imported successfully\")\n",
        "except ImportError as e:\n",
        "    PHASE4_AVAILABLE = False\n",
        "    print(f\"⚠️ Phase 4 components not found: {e}\")\n",
        "    print(\"   Verification features will be disabled.\")\n"
    ]
    
    fixed = False
    for cell in cells:
        if cell["cell_type"] == "code":
            source_str = "".join(cell["source"])
            if target_content in source_str or "aimo-solver-phase4" in source_str:
                print("Found import cell target. Updating...")
                cell["source"] = new_source
                fixed = True
                break
    
    if fixed:
        with open(NOTEBOOK_PATH, 'w') as f:
            json.dump(nb, f, indent=1)
        print("✅ Fixed notebook imports and restored PHASE4_AVAILABLE.")
    else:
        print("❌ Could not find import cell to fix.")

if __name__ == "__main__":
    fix_imports()

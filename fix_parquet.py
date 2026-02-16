import json
import os

NOTEBOOK_PATH = "notebooks/aimo3_kaggle_ready.ipynb"

def add_parquet_logic():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    # We need to add a cell at the end (or modify the save cell) to also save parquet
    # Let's find the "Save Submission" cell.
    
    # Logic to add:
    parquet_code = [
        "\n",
        "# COMPATIBILITY FIX: Generate submission.parquet if required\n",
        "try:\n",
        "    # Check for parquet dependencies\n",
        "    try:\n",
        "        import pyarrow\n",
        "    except ImportError:\n",
        "        try:\n",
        "            import fastparquet\n",
        "        except ImportError:\n",
        "            print(\"⚠️ No parquet engine found (pyarrow/fastparquet). Skipping parquet generation.\")\n",
        "            sub_df = None\n",
        "    \n",
        "    if 'sub_df' in locals() or os.path.exists('submission.csv') or 'submission_df' in locals():\n",
        "        df_to_save = None\n",
        "        if 'submission_df' in locals():\n",
        "            df_to_save = submission_df\n",
        "        elif os.path.exists('submission.csv'):\n",
        "            df_to_save = pd.read_csv('submission.csv')\n",
        "        \n",
        "        if df_to_save is not None and not df_to_save.empty:\n",
        "            try:\n",
        "                df_to_save.to_parquet('submission.parquet')\n",
        "                print(\"✅ Generated submission.parquet\")\n",
        "            except Exception as pe:\n",
        "                print(f\"⚠️ Parquet export failed (likely missing engine): {pe}\")\n",
        "        else:\n",
        "            print(\"⚠️ No data to save to parquet.\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Error in parquet generation block: {e}\")\n"
    ]
    
    # Find the cell that saves submission.csv or the end of the notebook
    # In my previous view, the save cell was conditional: "if not KAGGLE_MODE:"
    # I should verify this locally.
    
    # I'll just append a new cell at the very end to be safe.
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": parquet_code
    }
    
    nb["cells"].append(new_cell)
    print("Added parquet generation cell.")
    
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"✅ Updated {NOTEBOOK_PATH} with parquet logic.")

if __name__ == "__main__":
    add_parquet_logic()

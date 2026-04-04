import json

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\Model-tester.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source_code = [
    "# ── TEST 4: IMPUTED VS NON-IMPUTED LABELS ─────────────────────────────────────\n",
    "print(\"\\n============================================================\")\n",
    "print(\"  TEST 4 — REDUCED CHI-SQUARED (IMPUTED VS NATURAL)\")\n",
    "print(\"============================================================\")\n",
    "\n",
    "# First, construct the bad_mask for the test set\n",
    "with h5py.File(config.H5_PATH, 'r') as f:\n",
    "    get_col = lambda k: f['metadata'][k]\n",
    "    keys    = f['metadata'].dtype.names\n",
    "    \n",
    "    raw_values = np.stack([get_col(p)[config.TEST_START:test_end] for p in config.SELECTED_LABELS], axis=1)\n",
    "    test_bad_mask   = np.zeros_like(raw_values, dtype=bool)\n",
    "\n",
    "    for i, label in enumerate(config.SELECTED_LABELS):\n",
    "        flag_name = f\"{label}_FLAG\"\n",
    "        if flag_name in keys:\n",
    "            flg = get_col(flag_name)[config.TEST_START:test_end]\n",
    "            if flg.dtype.names:\n",
    "                flg = flg[flg.dtype.names[0]]\n",
    "            if flg.dtype.kind == 'V':\n",
    "                flg = flg.view('<i4')\n",
    "            test_bad_mask[:, i] = flg.astype(int) != 0\n",
    "        elif label in ['TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI']:\n",
    "            test_bad_mask[:, i] = raw_values[:, i] < -5000\n",
    "\n",
    "# For each label and each model\n",
    "print(f\"{'Label':<10s} | {'Model':<15s} | {'N (Natural)':>12s} | {'χ²_red (Nat)':>14s} | {'N (Imputed)':>12s} | {'χ²_red (Imp)':>14s}\")\n",
    "print(\"-\" * 85)\n",
    "\n",
    "for i, label in enumerate(config.SELECTED_LABELS):\n",
    "    is_imputed = test_bad_mask[:, i]\n",
    "    is_natural = ~is_imputed\n",
    "    \n",
    "    n_imp = np.sum(is_imputed)\n",
    "    n_nat = np.sum(is_natural)\n",
    "    \n",
    "    if n_imp == 0:\n",
    "        continue\n",
    "        \n",
    "    for model_name, chi2_array in model_chi2.items():\n",
    "        if len(chi2_array) == 0:\n",
    "            continue\n",
    "            \n",
    "        # calculate median\n",
    "        med_nat = float(np.median(chi2_array[is_natural])) if n_nat > 0 else np.nan\n",
    "        med_imp = float(np.median(chi2_array[is_imputed])) if n_imp > 0 else np.nan\n",
    "        \n",
    "        if model_name == list(model_chi2.items())[0][0]:\n",
    "            label_print = label\n",
    "        else:\n",
    "            label_print = \"\"\n",
    "            \n",
    "        print(f\"{label_print:<10s} | {model_name:<15s} | {n_nat:>12d} | {med_nat:>14.3f} | {n_imp:>12d} | {med_imp:>14.3f}\")\n",
    "    print(\"-\" * 85)\n",
    "print(\"\\n✓ All tests complete.\")\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "imputation_test",
    "metadata": {},
    "outputs": [],
    "source": source_code
}

# we don't want to mess up the existing last cell's 'All tests complete', 
# actually wait, I just appended it to the new cell so I should remove the old one.
# Or just let it be. Just append.
nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Successfully appended new cell.")

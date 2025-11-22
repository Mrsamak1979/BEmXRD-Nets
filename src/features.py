import pandas as pd
import numpy as np
from pymatgen.core import Composition
from mendeleev import element as fetch_element

# ฟังก์ชันช่วยแปลงค่าให้ปลอดภัย
def safe_float(val):
    try:
        if isinstance(val, dict): return float(list(val.values())[0])
        if isinstance(val, (list, tuple)): return float(val[0])
        return float(val) if val is not None else 0.0
    except:
        return 0.0

def get_element_data(el_symbol):
    try:
        el = fetch_element(str(el_symbol).strip())
        
        first_ion = 0.0
        if el.ionenergies:
             if isinstance(el.ionenergies, dict): first_ion = el.ionenergies.get(1, 0.0)
             elif isinstance(el.ionenergies, list): first_ion = el.ionenergies[0]

        return {
            "AtomicMass": safe_float(el.atomic_weight),
            "AtomicNumber": safe_float(el.atomic_number),
            "AtomicRadius": safe_float(el.atomic_radius),
            "BoilingPoint": safe_float(el.boiling_point),
            "Density": safe_float(el.density),
            "Electronegativity": safe_float(el.en_pauling),
            "FirstIonization": safe_float(first_ion),
            "MassNumber": safe_float(el.mass_number),
            "MeltingPoint": safe_float(el.melting_point),
            "SpecificHeat": safe_float(el.specific_heat)
        }
    except:
        return {}

def calculate_row_features(chem_formula):
    try:
        if pd.isna(chem_formula) or str(chem_formula).strip() == "":
            return {}

        comp = Composition(str(chem_formula))
        elements = [str(e) for e in comp.elements]
        
        # --- จุดที่แก้ไข (FIXED HERE) ---
        # เปลี่ยนจาก .get_amount(e) เป็น [e]
        amounts = [comp[e] for e in comp.elements]
        # -------------------------------
        
        row = {}
        prefixes = ['A', 'B', 'C', 'D', 'E']
        prefix_map = {p: f"{p}_{s}" for p, s in zip(prefixes, ['k', 'l', 'm', 'n', 'p'])}
        props = ["AtomicMass", "AtomicNumber", "AtomicRadius", "BoilingPoint", "Density", 
                 "Electronegativity", "FirstIonization", "MassNumber", "MeltingPoint", "SpecificHeat"]

        el_cache = {}
        for i, prefix in enumerate(prefixes):
            col_amt = prefix_map[prefix]
            if i < len(elements):
                el_sym, amt = elements[i], amounts[i]
                data = get_element_data(el_sym)
                
                row[col_amt] = amt
                for p in props:
                    val = data.get(p, 0.0)
                    row[f"{prefix}_{p}"] = val
                    if p not in el_cache: el_cache[p] = []
                    el_cache[p].append((amt, val))
            else:
                row[col_amt] = 0
                for p in props: row[f"{prefix}_{p}"] = 0
        
        total_atoms = sum(row[prefix_map[p]] for p in prefixes)
        row["Total_Atoms"] = total_atoms

        for p in props:
            if p in el_cache and total_atoms > 0:
                w_avg = sum(a*v for a,v in el_cache[p]) / total_atoms
                row[f"Weighted_{p}"] = w_avg
                for pre in prefixes:
                    row[f"{pre}SubW{p}_square"] = (row[f"{pre}_{p}"] - w_avg)**2
            else:
                row[f"Weighted_{p}"] = 0
                for pre in prefixes: row[f"{pre}SubW{p}_square"] = 0

        a_amt = row.get('A_k', 0)
        others = sum(row.get(prefix_map[p], 0) for p in prefixes[1:])
        row["SubA_kB_lC_mD_nE_p_square"] = (a_amt - others)**2
        
        for p_den in prefixes[1:]:
            den = row.get(prefix_map[p_den], 0)
            row[f"DivA_k{prefix_map[p_den]}"] = (a_amt / den) if den > 0 else 0
        
        return row
    except Exception as e:
        return {}

def generate_advanced_atomic_features(df, formula_col='full_formula'):
    print("⏳ Generating Advanced Atomic Features...")
    if formula_col not in df.columns:
        alternatives = ['formula', 'pretty_formula']
        for alt in alternatives:
            if alt in df.columns:
                formula_col = alt
                break
        else:
            return pd.DataFrame()

    feats = [calculate_row_features(f) for f in df[formula_col]]
    return pd.DataFrame(feats).fillna(0)

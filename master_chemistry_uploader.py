
import io
import pandas as pd
import optimizer as opt

def load_master(uploaded_file):
    """Load the supplied optimizer master through the backend loader.

    Streamlit UploadedFile objects are converted to the dictionary shape
    expected by the Colab-derived backend loader.
    """
    if uploaded_file is None:
        raise ValueError("No Excel file supplied.")
    data = uploaded_file.getvalue()
    df = opt.load_master_chemistry_excel({uploaded_file.name: data})

    # Fines_Pct is an optional productivity/master-display field in the
    # latest productivity layer and is intentionally not a required chemistry
    # loader column.
    df["Fines_Pct"] = 0.0
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        for frame in sheets.values():
            if "Fines_Pct" in frame.columns and "Material" in frame.columns:
                f = frame[["Material", "Fines_Pct"]].copy()
                f["Material"] = f["Material"].astype(str).str.strip()
                f["Fines_Pct"] = pd.to_numeric(f["Fines_Pct"], errors="coerce").fillna(0.0)
                f = f.drop_duplicates("Material").set_index("Material")
                df["Fines_Pct"] = f["Fines_Pct"].reindex(df.index).fillna(0.0).values
                break
    except Exception:
        pass

    return df

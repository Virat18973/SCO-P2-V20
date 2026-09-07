# Hospet Sinter Burden Control — v37 Dashboard Package (Manual Scenario Fix)

This package uses the supplied 31-Aug-2026 Streamlit dashboard (`app (3).py`) and the supplied v37.0 Streamlit backend (`optimizer (3).py`).

## Important fix in this package
The Manual Burden Control failure occurred when `solve_manual_scenario()` assigned a user quantity directly into `Tech_Min` / `Tech_Max` on a DataFrame whose column dtype could be integer/object/mixed under newer Pandas versions.

The backend has been updated only in the Manual Burden Control data-handling layer:
- scenario DataFrame is deep-copied;
- `Tech_Min`, `Tech_Max`, and `Available_Tonnes` are normalised to numeric float columns before assignment;
- scalar assignments use `.at[...]`;
- duplicate material names are rejected with a clear message.

No LP objective, chemistry equation, mandate, technical limit rule, coke/thermal rule, availability rule, or other optimisation mathematics has been changed by this fix.

## Files
- `app.py` — Streamlit dashboard
- `optimizer.py` — supplied v37.0 backend with the Manual Burden Control dtype safeguard
- `master_chemistry_uploader.py` — Excel/master input helper
- `Master_Chemistry_Input_Template.xlsx` — master chemistry input template
- `requirements.txt` — deployment dependencies

## Deployment
Upload these files to the same GitHub repository with these exact names:

```text
app.py
optimizer.py
master_chemistry_uploader.py
Master_Chemistry_Input_Template.xlsx
requirements.txt
```

Then deploy/reboot the Streamlit app.

## Manual Burden Control workflow
1. Run the main optimiser.
2. The optimiser result becomes the theoretical baseline.
3. Open Manual Burden Control.
4. Change the desired material quantity/change.
5. Click `RECALCULATE PRACTICAL SCENARIO`.
6. The changed materials are fixed as practical constraints.
7. The backend re-optimises the remaining eligible materials.
8. The page displays baseline vs practical cost/burden and practical chemistry/compliance.

## Filename requirement
Do not upload the Python files as `app (3).py` or `optimizer (3).py`. The application imports the backend with:

```python
import optimizer as opt
```

Therefore the deployed backend must be named exactly `optimizer.py`.

# Clean WDM Retrieval Script

This script replaces the experimental notebook retrieval loop with a cleaner Python workflow.

## Files

- `wdm_retrieval_clean.py` - retrieves NLDAS/GLDAS data, computes `SOLR`, `WIND`, and `CLOU`, and writes to WDM.
- `test_wdm_retrieval_clean.py` - offline smoke tests for the core transformation helpers.
- `station_example.csv` - example station input file.

## What it fixes

- Keeps all WDM inputs as pandas objects with a `DatetimeIndex`
- Avoids passing `numpy.ndarray` into `wdm.csvtowdm`
- Computes `CLOU` from `SOLR` in a separate step instead of inside the `SOLR` conversion branch
- Handles NLDAS vector wind as the magnitude of `Wind_N` and `Wind_E`
- Reuses existing Earthdata auth files (`.netrc`, `.dodsrc`, `.urs_cookies`, optional `.edl_token`) before prompting for credentials

## Earthdata credential helper

`run_retrieval(...)` now calls `ensure_earthdata_credentials()` before the first LDAS request.

Behavior:

- If a usable `.netrc` already exists in the target/home directory, current working directory, or another supplied search directory, it is reused.
- If `.urs_cookies` or `.dodsrc` are missing, they are created automatically.
- If no usable `.netrc` is found, the user is prompted for their Earthdata username and password and a new `.netrc` is written.

Example direct usage:

```python
from wdm_retrieval_clean import ensure_earthdata_credentials

auth_info = ensure_earthdata_credentials()
print(auth_info["netrc_path"])
print(auth_info["prompted"])
```

## Example usage

```powershell
C:\Users\KVENABLE\miniforge3\envs\wdm\python.exe C:\Users\KVENABLE\NLDASV2_Giovanni_BASINS\test_wdm_retrieval_clean.py
C:\Users\KVENABLE\miniforge3\envs\wdm\python.exe C:\Users\KVENABLE\NLDASV2_Giovanni_BASINS\wdm_retrieval_clean.py --stations C:\Users\KVENABLE\NLDASV2_Giovanni_BASINS\station_example.csv --start-date 2025-01-01 --end-date 2025-01-03 --wdm-file C:\Users\KVENABLE\NLDASV2_Giovanni_BASINS\MetData_clean.wdm
```

## Input CSV columns

Required columns:

- `StationName`
- `Latitude`
- `Longitude`
- `TimeZoneAdjustment`


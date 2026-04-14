from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tsgettoolbox import tsgettoolbox as tsget
from wdmtoolbox import wdmtoolbox as wdm

from cloud_cover_timeseries_from_solar import cloud_cover_timeseries_from_solar


CONSTITUENTS = ["ATEMP", "PRECIP", "SOLR", "WIND", "CLOU"]

GLDAS_CONSTITUENT_DETAILS = {
    "PRECIP": ["Precipitation", "mm", "GLDAS2:GLDAS_NOAH025_3H_2_1_Rainf_f_tavg", "kg/m^2/s"],
    "ATEMP": ["Air Temperature", "Celsius", "GLDAS2:GLDAS_NOAH025_3H_2_1_Tair_f_inst", "K"],
    "SOLR": ["Shortwave radiation flux", "Langleys", "GLDAS2:GLDAS_NOAH025_3H_2_1_SWdown_f_tavg", "W m-2"],
    "WIND": ["Wind Speed", "m s-1", "GLDAS2:GLDAS_NOAH025_3H_2_1_Wind_f_inst", "m s-1"],
    "CLOU": ["Cloud Cover Estimate", "Cloud Fraction", "GLDAS2:GLDAS_NOAH025_3H_2_1_SWdown_f_tavg", "W m-2"],
}

NLDAS_CONSTITUENT_DETAILS = {
    "PRECIP": ["Precipitation", "mm", "NLDAS:NLDAS_FORA0125_H_2_0_Rainf", "mm"],
    "ATEMP": ["Air Temperature", "Celsius", "NLDAS:NLDAS_FORA0125_H_2_0_Tair", "K"],
    "SOLR": ["Shortwave radiation flux", "Langleys", "NLDAS:NLDAS_FORA0125_H_2_0_SWdown", "W m-2"],
    "WIND": [
        "Wind Speed",
        "m s-1",
        "NLDAS:NLDAS_FORA0125_H_2_0_Wind_N",
        "NLDAS:NLDAS_FORA0125_H_2_0_Wind_E",
        "m s-1",
    ],
    "CLOU": ["Cloud Cover Estimate", "Cloud Fraction", "NLDAS:NLDAS_FORA0125_H_2_0_SWdown", "W m-2"],
}

US_BOUNDS = {
    "north": 49.3457868,
    "south": 24.7433195,
    "west": -124.7844079,
    "east": -66.9513812,
}


@dataclass(frozen=True)
class Station:
    name: str
    latitude: float
    longitude: float
    timezone_adjustment: float


@dataclass(frozen=True)
class RetrievalResult:
    series: pd.Series
    variable_name: str
    timestep_hours: int
    description: str
    column_name: str


def validate_lat_lon(latitude: float, longitude: float) -> bool:
    return -90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180


def load_stations(csv_path: str | os.PathLike[str]) -> list[Station]:
    station_df = pd.read_csv(csv_path)
    required_columns = ["StationName", "Latitude", "Longitude", "TimeZoneAdjustment"]
    missing = [column for column in required_columns if column not in station_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stations: list[Station] = []
    for _, row in station_df.iterrows():
        if not validate_lat_lon(row["Latitude"], row["Longitude"]):
            raise ValueError(
                f"Invalid coordinates for station {row['StationName']}: "
                f"{row['Latitude']}, {row['Longitude']}"
            )
        stations.append(
            Station(
                name=str(row["StationName"]),
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                timezone_adjustment=float(row["TimeZoneAdjustment"]),
            )
        )
    return stations


def in_contiguous_usa(station: Station) -> bool:
    return (
        US_BOUNDS["south"] < station.latitude < US_BOUNDS["north"]
        and US_BOUNDS["west"] < station.longitude < US_BOUNDS["east"]
    )


def get_catalog(station: Station) -> tuple[dict[str, list[str]], int, str]:
    if in_contiguous_usa(station):
        return NLDAS_CONSTITUENT_DETAILS, 1, "NLDAS"
    return GLDAS_CONSTITUENT_DETAILS, 3, "GLDAS"


def normalize_series(data: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 0:
            raise ValueError("Retrieved DataFrame has no columns.")
        series = data.iloc[:, 0].copy()
    elif isinstance(data, pd.Series):
        series = data.copy()
    else:
        raise TypeError(f"Expected pandas Series/DataFrame, got {type(data).__name__}")

    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna().sort_index()
    series.name = name
    return series


def convertunitforHSPF(constituent: str, series: pd.Series, ldas_var: str) -> pd.Series:
    series = normalize_series(series, constituent)

    if constituent == "ATEMP":
        series = series - 273.15
    elif constituent == "SOLR":
        series = series * 0.0864
    elif constituent == "PRECIP" and "GLDAS" in ldas_var:
        series = series * 3600 * 3

    series.name = constituent
    return series


def compute_cloud_cover(solr_series: pd.Series, lat_deg: float) -> pd.Series:
    solr_series = normalize_series(solr_series, "SOLR")
    solr_series = convertunitforHSPF("SOLR", solr_series, "SOLR")
    clou_df = cloud_cover_timeseries_from_solar(
        solr_values=solr_series.tolist(),
        dates=solr_series.index.to_pydatetime().tolist(),
        lat_deg=lat_deg,
    )
    clou_series = normalize_series(clou_df, "CLOU")
    clou_series.name = "CLOU"
    return clou_series


def fetch_ldas_series(station: Station, variable_name: str, start_date: str, end_date: str) -> pd.Series:
    result = tsget.ldas(
        lat=station.latitude,
        lon=station.longitude,
        variables=variable_name,
        startDate=start_date,
        endDate=end_date,
    )
    return normalize_series(result, variable_name)


def build_result_for_constituent(
    station: Station,
    constituent: str,
    start_date: str,
    end_date: str,
) -> RetrievalResult:
    catalog, timestep_hours, dataset_name = get_catalog(station)

    if constituent == "WIND" and dataset_name == "NLDAS":
        wind_n_var = catalog["WIND"][2]
        wind_e_var = catalog["WIND"][3]
        wind_n = fetch_ldas_series(station, wind_n_var, start_date, end_date)
        wind_e = fetch_ldas_series(station, wind_e_var, start_date, end_date)
        wind_n, wind_e = wind_n.align(wind_e, join="inner")
        wind_speed = np.sqrt(wind_n.pow(2) + wind_e.pow(2))
        wind_speed = normalize_series(wind_speed, "WIND")
        return RetrievalResult(
            series=wind_speed,
            variable_name=f"{wind_n_var} + {wind_e_var}",
            timestep_hours=timestep_hours,
            description="Vector Wind Speed",
            column_name="WIND",
        )

    if constituent == "CLOU":
        solar_var = catalog["CLOU"][2]
        solr_series = fetch_ldas_series(station, solar_var, start_date, end_date)
        clou_series = compute_cloud_cover(solr_series, station.latitude)
        return RetrievalResult(
            series=clou_series,
            variable_name=solar_var,
            timestep_hours=timestep_hours,
            description=solar_var,
            column_name="CLOU",
        )

    variable_name = catalog[constituent][2]
    series = fetch_ldas_series(station, variable_name, start_date, end_date)
    series = convertunitforHSPF(constituent, series, variable_name)
    description = "Vector Wind Speed" if constituent == "WIND" and dataset_name == "NLDAS" else variable_name
    return RetrievalResult(
        series=series,
        variable_name=variable_name,
        timestep_hours=timestep_hours,
        description=description,
        column_name=constituent,
    )


def prepare_wdm_input(series: pd.Series, constituent: str) -> pd.DataFrame:
    series = normalize_series(series, constituent)
    if series.empty:
        raise ValueError(f"No data available to write for constituent {constituent}.")
    return series.to_frame(name=constituent)


def write_result_to_wdm(
    wdm_path: str | os.PathLike[str],
    dsn: int,
    station: Station,
    constituent: str,
    result: RetrievalResult,
) -> None:
    input_df = prepare_wdm_input(result.series, constituent)
    wdm.createnewdsn(
        str(wdm_path),
        dsn,
        constituent=constituent,
        scenario="OBSERVED",
        location=station.name[:8],
        tcode=3,
        statid=station.name,
        tsstep=result.timestep_hours,
        description=result.description,
    )
    wdm.csvtowdm(str(wdm_path), dsn, input_ts=input_df)


def derive_log_path(wdm_path: str | os.PathLike[str], explicit_log_path: str | None = None) -> Path:
    if explicit_log_path:
        return Path(explicit_log_path)
    wdm_path = Path(wdm_path)
    return wdm_path.with_name(wdm_path.stem.replace("MetData", "MetLog") + ".txt")


def run_retrieval(
    stations: Iterable[Station],
    start_date: str,
    end_date: str,
    wdm_path: str | os.PathLike[str],
    log_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    wdm_path = Path(wdm_path).resolve()
    log_path = derive_log_path(wdm_path, None if log_path is None else str(log_path)).resolve()

    wdm.createnewwdm(str(wdm_path), overwrite=True)
    dsn = 1

    with open(log_path, "w", encoding="utf-8") as logfile:
        logfile.write(
            "Started downloading data at "
            f"{datetime.now().isoformat()} and saving in {wdm_path.name}\n"
        )

        for station in stations:
            logfile.write(
                f"Station: {station.name}, Latitude: {station.latitude}, "
                f"Longitude: {station.longitude}, TimeZoneAdjustment: {station.timezone_adjustment}\n"
            )

            for constituent in CONSTITUENTS:
                result = build_result_for_constituent(
                    station=station,
                    constituent=constituent,
                    start_date=start_date,
                    end_date=end_date,
                )
                write_result_to_wdm(wdm_path, dsn, station, constituent, result)
                logfile.write(
                    f"Constituent: {constituent}, Column Name: {result.column_name}, DSN: {dsn}\n"
                )
                dsn += 1

    return wdm_path, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve NLDAS/GLDAS time series and write them to a WDM file."
    )
    parser.add_argument(
        "--stations",
        default="station_example.csv",
        help="CSV file with StationName, Latitude, Longitude, TimeZoneAdjustment columns.",
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--wdm-file", default="MetData_clean.wdm", help="Output WDM file path.")
    parser.add_argument("--log-file", default=None, help="Optional explicit MetLog output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_stations(args.stations)
    wdm_path, log_path = run_retrieval(
        stations=stations,
        start_date=args.start_date,
        end_date=args.end_date,
        wdm_path=args.wdm_file,
        log_path=args.log_file,
    )
    print(f"Created WDM file: {wdm_path}")
    print(f"Created log file: {log_path}")


if __name__ == "__main__":
    main()



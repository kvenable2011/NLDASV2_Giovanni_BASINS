import tempfile
from pathlib import Path

import pandas as pd

import wdm_retrieval_clean as wrc
from wdm_retrieval_clean import (
	Station,
	build_result_for_constituent,
	compute_cloud_cover,
	compute_dewpoint,
	convertunitforHSPF,
	ensure_earthdata_credentials,
	prepare_wdm_input,
)


def run_smoke_tests() -> None:
	index = pd.date_range("2025-01-01", periods=3, freq="D")

	atemp = pd.Series([273.15, 274.15, 275.15], index=index, name="ATEMP")
	atemp_c = convertunitforHSPF("ATEMP", atemp, "NLDAS:NLDAS_FORA0125_H_2_0_Tair")
	assert round(float(atemp_c.iloc[0]), 6) == 0.0

	solr = pd.Series([500.0, 600.0, 700.0], index=index, name="SOLR")
	solr_lang = convertunitforHSPF("SOLR", solr, "NLDAS:NLDAS_FORA0125_H_2_0_SWdown")
	assert round(float(solr_lang.iloc[0]), 6) == round(500.0 * 0.0864, 6)

	clou = compute_cloud_cover(solr, lat_deg=40.0)
	assert isinstance(clou, pd.Series)
	assert isinstance(clou.index, pd.DatetimeIndex)
	assert clou.name == "CLOU"
	assert len(clou) == 3

	spec_humidity = pd.Series([0.007, 0.008, 0.009], index=index, name="SPCHUM")
	dewpoint_air_temp = pd.Series([293.15, 294.15, 295.15], index=index, name="ATEMP")
	dewpoint = compute_dewpoint(spec_humidity, dewpoint_air_temp)
	assert isinstance(dewpoint, pd.Series)
	assert isinstance(dewpoint.index, pd.DatetimeIndex)
	assert dewpoint.name == "DEWP"
	assert round(float(dewpoint.iloc[0]), 6) == 49.491792
	assert len(dewpoint) == 3

	wdm_df = prepare_wdm_input(clou, "CLOU")
	assert list(wdm_df.columns) == ["CLOU"]
	assert isinstance(wdm_df.index, pd.DatetimeIndex)

	original_fetch_ldas_series = wrc.fetch_ldas_series
	try:
		def fake_fetch_ldas_series(station, variable_name, start_date, end_date):
			data_map = {
				"NLDAS:NLDAS_FORA0125_H_2_0_Qair": pd.Series([0.007, 0.008, 0.009], index=index),
				"NLDAS:NLDAS_FORA0125_H_2_0_Tair": pd.Series([293.15, 294.15, 295.15], index=index),
				"NLDAS:NLDAS_FORA0125_H_2_0_PSurf": pd.Series([101325.0, 101325.0, 101325.0], index=index),
			}
			return data_map[variable_name]

		wrc.fetch_ldas_series = fake_fetch_ldas_series
		station = Station("CentralPark", 40.78, -73.97, -5)
		dewp_result = build_result_for_constituent(
			station=station,
			constituent="DEWP",
			start_date="2025-01-01",
			end_date="2025-01-03",
		)
		assert dewp_result.column_name == "DEWP"
		assert dewp_result.series.name == "DEWP"
		assert len(dewp_result.series) == 3
		assert round(float(dewp_result.series.iloc[0]), 6) == 49.491792
	finally:
		wrc.fetch_ldas_series = original_fetch_ldas_series

	with tempfile.TemporaryDirectory() as temp_dir_name:
		temp_dir = Path(temp_dir_name)
		source_dir = temp_dir / "source"
		target_dir = temp_dir / "target"
		source_dir.mkdir()
		target_dir.mkdir()
		(source_dir / ".netrc").write_text(
			"machine urs.earthdata.nasa.gov login saved_user password saved_pass",
			encoding="utf-8",
		)
		(source_dir / ".edl_token").write_text("cached-token", encoding="utf-8")

		auth_info = ensure_earthdata_credentials(
			target_dir=target_dir,
			search_dirs=[source_dir],
			prompt_if_missing=False,
			include_default_search_dirs=False,
		)
		assert auth_info["prompted"] is False
		assert auth_info["username"] == "saved_user"
		assert (target_dir / ".netrc").exists()
		assert (target_dir / ".dodsrc").exists()
		assert (target_dir / ".urs_cookies").exists()
		assert (target_dir / ".edl_token").read_text(encoding="utf-8") == "cached-token"

	with tempfile.TemporaryDirectory() as temp_dir_name:
		target_dir = Path(temp_dir_name)
		auth_info = ensure_earthdata_credentials(
			target_dir=target_dir,
			search_dirs=[target_dir],
			prompt_if_missing=True,
			include_default_search_dirs=False,
			input_func=lambda prompt: "prompted_user",
			password_func=lambda prompt: "prompted_pass",
		)
		assert auth_info["prompted"] is True
		assert auth_info["username"] == "prompted_user"
		netrc_text = (target_dir / ".netrc").read_text(encoding="utf-8")
		assert "prompted_user" in netrc_text
		assert "prompted_pass" in netrc_text

	print("All smoke tests passed.")


if __name__ == "__main__":
	run_smoke_tests()


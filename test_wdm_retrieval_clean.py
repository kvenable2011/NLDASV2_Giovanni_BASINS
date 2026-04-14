import pandas as pd

from wdm_retrieval_clean import compute_cloud_cover, convertunitforHSPF, prepare_wdm_input


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

	wdm_df = prepare_wdm_input(clou, "CLOU")
	assert list(wdm_df.columns) == ["CLOU"]
	assert isinstance(wdm_df.index, pd.DatetimeIndex)

	print("All smoke tests passed.")


if __name__ == "__main__":
	run_smoke_tests()


import math
from datetime import datetime

def get_julian_day_from_date(date_obj):
    """
    Convert a datetime object to Julian day of the year (1-366).
    """
    return date_obj.timetuple().tm_yday

def process_clou_from_solar(solr_values, dates):
    """
    Calculate cloud cover (CLOU) from solar radiation (SOLR) values.
    Args:
        solr_values (list or array): List of solar radiation values.
        dates (list of datetime): List of datetime objects corresponding to solr_values.
    Returns:
        list: Calculated cloud cover values (CLOU).
    """
    clou_values = []
    for idx, solr in enumerate(solr_values):
        # Julian day calculation (not strictly needed for this formula, but included for completeness)
        julian_day = get_julian_day_from_date(dates[idx])
        if solr > 990:
            clou = 0.0
        elif solr < 247.5:
            clou = 10.0
        else:
            clou = (((990.0 - solr) / 742.5) ** (1 / 3)) * 10.0
        clou_values.append(clou)
    return clou_values

# Example usage:
if __name__ == "__main__":
    # Example SOLR values and dates
    solr_values = [1000, 500, 200, 800]
    dates = [datetime(2026, 4, 8), datetime(2026, 4, 9), datetime(2026, 4, 10), datetime(2026, 4, 11)]
    clou = process_clou_from_solar(solr_values, dates)
    print("CLOU values:", clou)


from datetime import datetime
from typing import List

# Degrees to radians constant
DegreesToRadians = 0.01745329252

# X1 array (1-based in .NET, so Python index 0 is 0, index 1 is 10.00028, etc.)
X1 = [
    0, 10.00028, 41.0003, 69.22113, 100.5259, 130.8852, 161.2853,
    191.7178, 222.1775, 253.66, 281.1629, 309.6838, 341.221
]

# c array (12x13, 1-based in .NET, so Python index 0 is row 0, etc.)
c = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4.0, 2.0, -1.5, -3.0, -2.0, 1.0, 3.0, 2.5, 1.0, 1.0, 2.0, 1.0],
    [0, 3.0, 4.0, 0.0, -3.0, -2.5, 0.0, 2.0, 3.0, 2.0, 1.5, 2.0, 1.0],
    [0, 0.0, 3.5, 1.5, -1.0, -2.0, -1.0, 1.5, 3.0, 3.0, 1.5, 2.0, 1.0],
    [0, -2.0, 2.5, 3.5, 0.0, -2.0, -1.0, 0.5, 3.0, 3.0, 2.0, 2.0, 1.0],
    [0, -4.0, 0.5, 3.0, 1.0, -0.5, -1.0, 0.0, 2.0, 2.5, 2.5, 2.0, 1.0],
    [0, -5.0, -1.5, 2.0, 3.0, 0.5, -1.0, -0.5, 1.0, 2.5, 2.5, 2.0, 1.0],
    [0, -5.0, -3.5, 1.0, 3.0, 1.5, 0.0, -0.5, 1.0, 2.0, 2.0, 2.0, 1.0],
    [0, -4.0, -4.5, -1.0, 2.5, 3.0, 1.0, 0.0, 0.0, 1.5, 2.0, 2.0, 1.0],
    [0, -2.0, -4.0, -3.0, 1.0, 3.0, 2.0, 0.5, 0.0, 1.5, 2.0, 1.0, 1.0],
    [0, 0.0, -3.5, -4.0, -0.5, 3.0, 3.0, 1.5, 1.0, 1.0, 2.0, 1.0, 1.0]
]

# XLax array (45x7, 1-based in .NET, so Python index 0 is row 0, etc.)
XLax = [
    [-9, -9, -9, -9, -9, -9, -9],  # 0
    [-9, -9, -9, -9, -9, -9, -9],  # 1
    [-9, -9, -9, -9, -9, -9, -9],  # 2
    [-9, -9, -9, -9, -9, -9, -9],  # 3
    [-9, -9, -9, -9, -9, -9, -9],  # 4
    [-9, -9, -9, -9, -9, -9, -9],  # 5
    [-9, -9, -9, -9, -9, -9, -9],  # 6
    [-9, -9, -9, -9, -9, -9, -9],  # 7
    [-9, -9, -9, -9, -9, -9, -9],  # 8
    [-9, -9, -9, -9, -9, -9, -9],  # 9
    [-9, -9, -9, -9, -9, -9, -9],  # 10
    [-9, -9, -9, -9, -9, -9, -9],  # 11
    [-9, -9, -9, -9, -9, -9, -9],  # 12
    [-9, -9, -9, -9, -9, -9, -9],  # 13
    [-9, -9, -9, -9, -9, -9, -9],  # 14
    [-9, -9, -9, -9, -9, -9, -9],  # 15
    [-9, -9, -9, -9, -9, -9, -9],  # 16
    [-9, -9, -9, -9, -9, -9, -9],  # 17
    [-9, -9, -9, -9, -9, -9, -9],  # 18
    [-9, -9, -9, -9, -9, -9, -9],  # 19
    [-9, -9, -9, -9, -9, -9, -9],  # 20
    [-9, -9, -9, -9, -9, -9, -9],  # 21
    [-9, -9, -9, -9, -9, -9, -9],  # 22
    [-9, -9, -9, -9, -9, -9, -9],  # 23
    [-9, 616.17, -147.83, -27.17, -3.17, 11.84, 2.02],  # 24
    [-9, 609.97, -154.71, -27.49, -2.97, 12.04, 1.3],   # 25
    [-9, 603.69, -161.55, -27.69, -2.78, 12.22, 0.64],  # 26
    [-9, 597.29, -168.33, -27.78, -2.6, 12.38, 0.02],   # 27
    [-9, 590.81, -175.05, -27.74, -2.43, 12.53, -0.56], # 28
    [-9, 584.21, -181.72, -27.57, -2.28, 12.67, -1.1],  # 29
    [-9, 577.53, -188.34, -27.29, -2.14, 12.8, -1.6],   # 30
    [-9, 570.73, -194.91, -26.89, -2.02, 12.92, -2.05], # 31
    [-9, 563.85, -201.42, -26.37, -1.91, 13.03, -2.45], # 32
    [-9, 556.85, -207.29, -25.72, -1.81, 13.13, -2.8],  # 33
    [-9, 549.77, -214.29, -24.96, -1.72, 13.22, -3.1],  # 34
    [-9, 542.57, -220.65, -24.07, -1.64, 13.3, -3.35],  # 35
    [-9, 535.3, -226.96, -23.07, -1.59, 13.36, -3.58],  # 36
    [-9, 527.9, -233.22, -21.95, -1.55, 13.4, -3.77],   # 37
    [-9, 520.44, -239.43, -20.7, -1.52, 13.42, -3.92],  # 38
    [-9, 512.84, -245.59, -19.33, -1.51, 13.42, -4.03], # 39
    [-9, 505.19, -251.69, -17.83, -1.51, 13.41, -4.1],  # 40
    [-9, 497.4, -257.74, -16.22, -1.52, 13.39, -4.13],  # 41
    [-9, 489.52, -263.74, -14.49, -1.54, 13.36, -4.12], # 42
    [-9, 481.53, -269.7, -12.63, -1.57, 13.32, -4.07],  # 43
    [-9, 473.45, -275.6, -10.65, -1.63, 13.27, -3.98],  # 44
    [-9, 465.27, -281.45, -8.55, -1.71, 13.21, -3.85],  # 45
    [-9, 456.99, -287.25, -6.33, -1.8, 13.14, -3.68],   # 46
    [-9, 448.61, -292.99, -3.98, -1.9, 13.07, -3.47],   # 47
    [-9, 440.14, -298.68, -1.51, -2.01, 13.0, -3.3],    # 48
    [-9, 431.55, -304.32, 1.08, -2.13, 12.92, -3.17],   # 49
    [-9, 431.55, -304.32, 1.08, -2.13, 12.92, -3.17]    # 50
]


def get_julian_day_from_date(date_obj):
    """
    Convert a datetime object to Julian day of the year (1-366).
    """
    return date_obj.timetuple().tm_yday

class TimeSeries:
    def __init__(self, values: List[float], dates: List[datetime], constituent_name: str):
        self.values = values
        self.dates = dates
        self.constituent = constituent_name

def cloud_cover_value_from_solar(lat_deg, sol_rad, month, day):
    """
    Placeholder for the actual cloud cover calculation from solar radiation.
    Replace with the correct formula as needed.
    """
    # Example: Use the same formula as process_clou_from_solar for demonstration
    if sol_rad > 990:
        return 0.0
    elif sol_rad < 247.5:
        return 10.0
    else:
        return (((990.0 - sol_rad) / 742.5) ** (1 / 3)) * 10.0

import math

# Example: You must define these arrays with the same values as in your .NET code!
# X1 = [...]
# XLax = [...]
# c = [...]
# DegreesToRadians = 0.01745329252

def cloud_cover_value_from_solar2(aDegLat, aDayRad, aMon, aDay):
    """
    Compute daily cloud cover value from solar radiation.
    Args:
        aDegLat (float): Latitude in degrees.
        aDayRad (float): Daily solar radiation value.
        aMon (int): Month (1-12).
        aDay (int): Day of month.
    Returns:
        float: Cloud cover value (0-10).
    """
    lLatInt = int(math.floor(aDegLat))
    # Clamp latitude to valid range for XLax
    if lLatInt < 24:
        lLatInt = 24
    elif lLatInt >= len(XLax) - 1:
        lLatInt = len(XLax) - 2
    lLatFrac = aDegLat - lLatInt
    if lLatFrac <= 0.0001:
        lLatFrac = 0.0

    # Clamp month to valid range for X1 and c
    if aMon < 1:
        aMon = 1
    elif aMon > 12:
        aMon = 12

    # Interpolate coefficients from XLax
    A0 = XLax[lLatInt][1] + lLatFrac * (XLax[lLatInt + 1][1] - XLax[lLatInt][1])
    A1 = XLax[lLatInt][2] + lLatFrac * (XLax[lLatInt + 1][2] - XLax[lLatInt][2])
    A2 = XLax[lLatInt][3] + lLatFrac * (XLax[lLatInt + 1][3] - XLax[lLatInt][3])
    A3 = XLax[lLatInt][4] + lLatFrac * (XLax[lLatInt + 1][4] - XLax[lLatInt][4])
    b1 = XLax[lLatInt][5] + lLatFrac * (XLax[lLatInt + 1][5] - XLax[lLatInt][5])
    b2 = XLax[lLatInt][6] + lLatFrac * (XLax[lLatInt + 1][6] - XLax[lLatInt][6])

    b = aDegLat - 44.0
    a = aDegLat - 25.0
    Exp1 = 0.7575 - 0.0018 * a
    Exp2 = 0.725 + 0.00288 * b
    Lat1 = 2.139 + 0.0423 * a
    Lat2 = 30.0 - 0.667 * a
    Lat3 = 2.9 - 0.0629 * b
    Lat4 = 18.0 + 0.833 * b

    x = X1[aMon] + aDay
    x *= DegreesToRadians

    Y100 = A0 + A1 * math.cos(x) + A2 * math.cos(2 * x) + A3 * math.cos(3 * x) + b1 * math.sin(x) + b2 * math.sin(2 * x)
    YRD = (aDayRad / Y100) * 100

    ii = int(math.ceil((min(100, YRD) + 10.0) / 10.0))
    if ii < 11:
        YRD = YRD - c[ii][aMon]

    if aDegLat > 43.0:
        try:
            ss = math.pow(((YRD - Lat4) / Lat3), 1 / Exp2)
        except ValueError:
            ss = 0.0
    else:
        try:
            ss = math.pow(((YRD - Lat2) / Lat1), 1 / Exp1)
        except ValueError:
            ss = 0.0

    if ss < 0.0:
        ss = 0.0

    # get cloud cover from %sun
    base = -((ss / 100) - 1)
    if base <= 0:
        clou = 0.0
    else:
        clou = 10 * math.pow(base, 3 / 5)

    return clou
import pandas as pd
def cloud_cover_timeseries_from_solar(solr_values, dates, lat_deg):
    """
    Compute daily cloud cover timeseries from daily solar radiation values.
    Args:
        solr_values (list of float): Solar radiation values (Langleys).
        dates (list of datetime): Dates corresponding to the values.
        lat_deg (float): Latitude in degrees.
    Returns:
        pd.DataFrame: DataFrame with columns 'date' and 'CLOU'.
    """
    clou_values = []
    for idx, solr in enumerate(solr_values):
        date = dates[idx]
        month = date.month
        day = date.day
        clou = cloud_cover_value_from_solar2(lat_deg, solr, month, day)
        clou_values.append(clou)
    # Return as DataFrame
    df = pd.DataFrame({'CLOU': clou_values}, index=dates)
    #df.index.name = 'date'
    return df
# Example usage:
if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime

    # Example solar radiation values and corresponding dates
    solr_values = [1000, 800, 500, 300, 200]
    dates = [
        datetime(2026, 3, 23),
        datetime(2026, 3, 24),
        datetime(2026, 3, 25),
        datetime(2026, 3, 26),
        datetime(2026, 3, 27)
    ]
    lat_deg = 40.0

    # Call the function to get a DataFrame
    clou_df = cloud_cover_timeseries_from_solar(solr_values, dates, lat_deg)

    # Print the resulting DataFrame
    print("CLOU DataFrame:")
    print(clou_df)


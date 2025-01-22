import pandas as pd
import pycountry


def iso3_to_iso2(iso3_code):
    # Iterate through countries in pycountry and find a match for the ISO3 code
    country = pycountry.countries.get(alpha_3=iso3_code)
    if country:
        return country.alpha_2
    else:
        return None  # Return None if no match is found
    

def generate_date_ranges(start_dt, num_weeks=4):
    start_dt = pd.to_datetime(start_dt)
    date_ranges = []
    for i in range(num_weeks):
        start = start_dt + pd.Timedelta(weeks=i)
        end = start + pd.Timedelta(weeks=1)
        date_ranges.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    return date_ranges
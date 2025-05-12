import requests
import pandas as pd
from time import sleep

def get_weather_forecast():
    # API token from https://www.ncdc.noaa.gov/cdo-web/webservices/v2
    #  National Centers For Environmental Information (get historical weather)
    TOKEN = "DrNWqvsREOchlHDOGihmOVfcRYzWCBgl"
    headers = {"token": TOKEN}

    years = range(2019, 2024)
    records = []

    for year in years:
        print(f"Fetching data for {year}...")
        year_data = {}

        # Try TAVG
        params = {
            "datasetid": "GHCND",
            "stationid": "GHCND:USW00012815",  # Eglin AFB (also in Okaloosa)
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "datatypeid": "TAVG",
            "units": "metric",
            "limit": 1000
        }
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        resp = requests.get(url, headers=headers, params=params)

        if resp.status_code == 200 and "results" in resp.json():
            df = pd.DataFrame(resp.json()["results"])
            if not df.empty:
                year_data["TEMPERATURE"] = df["value"].mean()
        else:
            print(f"TAVG not found for {year}, trying TMAX/TMIN...")

        # Compute TAVG if needed
        if "TEMPERATURE" not in year_data:
            temps = {}
            for dtype in ["TMAX", "TMIN"]:
                params["datatypeid"] = dtype
                resp = requests.get(url, headers=headers, params=params)
                if resp.status_code == 200 and "results" in resp.json():
                    df = pd.DataFrame(resp.json()["results"])
                    if not df.empty:
                        temps[dtype] = df["value"].mean()
                sleep(1)

            if "TMAX" in temps and "TMIN" in temps:
                year_data["TEMPERATURE"] = (temps["TMAX"] + temps["TMIN"]) / 2

        # AWND (wind speed)
        params["datatypeid"] = "AWND"
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200 and "results" in resp.json():
            df = pd.DataFrame(resp.json()["results"])
            if not df.empty:
                year_data["WINDSPEED"] = df["value"].mean()

        # If complete, convert temp to Fahrenheit and record
        if "TEMPERATURE" in year_data and "WINDSPEED" in year_data:
            temp_f = (year_data["TEMPERATURE"] * 9/5) + 32  # C to F
            records.append({
                "YEAR_": year,
                "TEMPERATURE": round(temp_f, 2),
                "WINDSPEED": round(year_data["WINDSPEED"], 2)
            })
        else:
            print(f"Incomplete data for {year}: {year_data}")

        sleep(1)

    df_final = pd.DataFrame(records)
    df_final.to_csv("okaloosa_weather_by_year.csv", index=False)
    print("Saved to 'okaloosa_weather_by_year.csv'")
    return df_final

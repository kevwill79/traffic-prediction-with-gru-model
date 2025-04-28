# -*- coding: utf-8 -*-
"""
Created on Tue Apr 1 22:54:36 2025

@author: kwill
"""

import pandas as pd
import PyPDF2
import re

def get_historical_traffic(csv_path):
    
    # Read historical data and return all data from Okaloosa County FL
    df = pd.read_csv(csv_path)
    #filtered_df = df[df['COUNTY'] == 'Okaloosa']
    #df = filtered_df[['YEAR_', 'ROADWAY', 'DESC_FRM', 'DESC_TO', 'AADT']]
        
    return df

# Extract text from the CSV file
def extract_traffic_data(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    lines = text.split("\n")
    data = []
    
    # To remove the date from the df
    regex_pattern = r'\b(201[0-9]|202[0-5])\b'
    
    for line in lines:
        parts = line.split()
        if parts[0].isdigit():  # Ensuring the first part is a site number
            try:
                site = parts[0]
                aadt = int(re.sub(r'\D', '', parts[5]))  # Extracting numeric AADT value
#                data.append([site, aadt])
#            except ValueError:
#                continue  # Skip lines with invalid data
    
#    df = pd.DataFrame(data, columns=["Site", "AADT"])
#    return df

                # Get full description (everything between SITE and direction)
                desc_start = line.find(parts[1]) + len(parts[1])
                desc_end = line.lower().find('n ') if 'n ' in line.lower() else line.lower().find('e ')
                description = line[desc_start:desc_end].strip()

                # Filter by target keywords
                #if any(kw in description.upper() for kw in ["SR 85", "SR 285", "US 98", "SR 98", "HWY 98"]):
                data.append([site, description, aadt])
            except Exception:
                continue

    df = pd.DataFrame(data, columns=["Site", "Description", "AADT"])
    return df

test_df = extract_traffic_data("C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/fdot_historical_traffic_data/3_57_CAADT.pdf")
test_df2 = get_historical_traffic("C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/fdot_historical_traffic_data/Annual_Average_Daily_Traffic_Historical_TDA.csv")
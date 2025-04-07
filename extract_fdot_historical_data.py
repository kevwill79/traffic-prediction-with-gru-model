# -*- coding: utf-8 -*-
"""
Created on Sat Apr 5 22:54:36 2025

@author: kwill
"""

import pandas as pd
import PyPDF2
import re

# Extract text from the PDF file
def extract_traffic_data(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    lines = text.split("\n")
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit():  # Ensuring the first part is a site number
            try:
                site = parts[0]
                aadt = int(re.sub(r'\D', '', parts[5]))  # Extracting numeric AADT value
                data.append([site, aadt])
            except ValueError:
                continue  # Skip lines with invalid data
    
    df = pd.DataFrame(data, columns=["Site", "AADT"])
    return df
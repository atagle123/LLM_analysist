import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
#import openpyxl 



def prompt_building_industry(industry="Enaex",ratios_path="data\outputs\SIGDO KOPPERS S.A. Final.xlsx"):  # enaex
    sheet_name = f'Output {industry}'

    # Read the Excel file
    df = pd.read_excel(ratios_path, sheet_name=sheet_name)

    return(df)


if __name__=="__main__":
    df=prompt_building_industry(industry="Enaex")
    print(df)
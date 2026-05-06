import pandas as pd
import os

base_path = r"C:\Users\Cypress\Desktop\data1.0"

df = pd.read_stata(os.path.join(base_path, "Sample_Infor.dta"))

print(df.columns.tolist())
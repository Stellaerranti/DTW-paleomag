import numpy as np
import pandas as pd

xl_file = pd.ExcelFile('I_full data.xlsx')

dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
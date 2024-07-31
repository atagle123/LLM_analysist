import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import json
from llama_index.core import PromptTemplate
from utils.json_utils import read_json

class Prompt_builder:
    def __init__(self,max_context=16000) -> None:
        self.prompt=""
        self.max_context=max_context
        pass

    def add_prompt(self,prompt):
        prompt_lenght=len(self.prompt) + len(prompt) + 1
        if prompt_lenght > self.max_context:
            raise ValueError(f"Prompt length exceeds maximum context window by {prompt_lenght}.")
        self.prompt += "\n" + prompt

    def __call__(self):
        return(self.prompt)

    def add_bullet_point(self, text):
        self.prompt += f"\n- {text}"

    def add_numbered_point(self, text, number):
        self.prompt += f"\n{number}. {text}"

    def export_to_json(self):
        import json
        return json.dumps({"prompt": self.prompt})
    
    def clear_prompt(self):
        self.prompt = ""

    def display_prompt(self):
        print(self.prompt)





class Industry_prompt(Prompt_builder):
    def __init__(self,industry="Enaex",ratios_path="data\outputs\SIGDO KOPPERS S.A. Final.xlsx",max_context=16000) -> None: # enaex, max context for gpt 3.5 
        super().__init__(max_context=max_context)
        self.industry=industry
        self.ratios_path=ratios_path
        pass

    def main_prompt_builder(self,ratios_year=2020):

        info_dict=read_json(file=f"src\prompts\{self.industry}_prompts.json") # change to self file
        self.add_prompt(f"You are an expert financial analyst and you have to provide and exhaustive analysis of the {self.industry} company, i will provide you information below:")
        self.add_prompt(f"Here i will display you all the relevant information about {self.industry} company.")
        self.add_prompt(info_dict["introduction"])
        ratios_prompt=self.prompt_ratios_building(ratios_year)
        self.add_prompt(ratios_prompt)



        return(self.prompt)



    def prompt_ratios_building(self,ratios_year,ratios_indexes=["Growth (Var. YoY)","Margins","Liquidity & leverage metrics","Turnover metrics","Profitability metrics"]):  
        sheet_name = f'Output {self.industry}'

        # Read the Excel file
        df = pd.read_excel(self.ratios_path, sheet_name=sheet_name,header=3,index_col=1)
        df=self.preprocess_output(df)

        df=df.iloc[:,4*(ratios_year-2017):] # asume que las fechas parten desde l 1q 2017, probablemente cambiar
        df=df.loc[ratios_indexes]
        df_json = df.to_json(double_precision=4,index=True,orient="index")
        json_object = json.loads(df_json)
        prompt=self.parse_json_to_prompt(json_object)

        return(prompt)

    def preprocess_output(self,df):

        df = df.dropna(axis=1, how='all') # Eliminar columnas completamente vacías
        df = df[df.index.notnull()] # Eliminar indices vacíos

        level = (~(df.isna().all(axis=1))).astype(int)
        df["Category"]=df.index.where(level==0)
        df["Category"].fillna(method='ffill', inplace=True)
        df["Metric"]=df.index.where(level==1)
        df.set_index(['Category', 'Metric'], inplace=True)  # set multiindex

        df = df.dropna(axis=0, how='all') # Eliminar filas completamente vacías
        df=df.round(1) #redondear para disminuir informacion no tan relevante y el tamaño del contexto
        return(df)

    def parse_json_to_prompt(self,json_object):
        prompt=f"Know i will provide you all the relevant metrics and information of the last periods for {self.industry}"
        for index, values_dict in json_object.items():
            index=eval(index)
            category_name=index[0]
            metric_name=index[1]
            json_str=self.json_parser(values_dict,index)
            intro_metric=f"Here you have information about {category_name} {metric_name} metric, dispayed for all quarters {json_str} \n" # maybe add with a dict information about each metric and what does it mean
            prompt+=intro_metric

        return(prompt)

    def json_parser(self,dict,index):

        if "%" in index[0] or "%" in index[1]:
            dict={key: self.format_percentage(value) for key, value in dict.items()}

        else:
            dict={key: self.format_json_values(value) for key, value in dict.items()} # ver el tema para los porcentajes... 

        json_str=json.dumps(dict, indent=0)
        json_str=json_str.replace('"', '')
        json_str=json_str.replace(':', '')
        json_str=json_str.replace(',', '')
        json_str=json_str.replace('{', '')
        json_str=json_str.replace('}', '')
        json_str=json_str.replace('null', '')
        return(json_str)
    
    def format_percentage(self,num):
        if num is None:
            return None  # Keep null values as is
        
        return(f"{num}%")

    def format_number(self,num):
        """Formats a number to a shorter string representation."""
        if num is None:
            return None  # Keep null values as is
        if abs(num) >= 1_000_000_000:  # Billions
            return f"{num / 1_000_000_000:.1f}B"
        
        elif abs(num) >= 100_000_000:  # Millions
            return f"{num / 1_000_000_000:.2f}B"
        
        elif abs(num) >= 1_000_000:  # Millions
            return f"{num / 1_000_000:.1f}M"

        elif abs(num) >= 100_000:  # Millions
            return f"{num / 1_000_000:.2f}M"
        
        elif abs(num) >= 1_000:  # Thousands
            return f"{num / 1_000:.1f}K"
        elif abs(num) >= 100:  # Hundreds
            return f"{num:.1f}"
        else:  # Below a hundred
            return str(num)  # Convert to string directly

    def format_json_values(self,data):
        """Recursively formats numeric values in JSON data."""
        if isinstance(data, dict):
            return {key: self.format_json_values(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.format_json_values(item) for item in data]
        else:
            return self.format_number(data)  # Format the number


        def build_introduction_to_industry_prompt(self,industry):
            raise NotImplementedError



if __name__=="__main__":
    industry_prompt=Industry_prompt(industry="Enaex")
    prompt=industry_prompt.main_prompt_builder()
    print(prompt)
    print(len(prompt))
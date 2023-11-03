import pandas as pd
import numpy as np
import re 
import time
import os

import cohere
from datetime import datetime

import tiktoken

import replicate
import streamlit as st

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens  

def add_file_to_master(master_file_list_path: str, new_dataset_name: str, file_path: str, new_dataset_description: str):
    file_list = pd.read_csv(master_file_list_path)
    file_list = pd.concat([pd.DataFrame([[new_dataset_name, 
                                          file_path, 
                                          new_dataset_description, 
                                          datetime.now()]], 
                                          columns = file_list.columns), file_list])
    file_list.to_csv(master_file_list_path, index = False)


def create_column_summary(in_path: str, file_type: str, out_dir: str, dataset_name: str):
    if file_type == ".csv":
        data = pd.read_csv(in_path)
    elif file_type == ".feather":
        data = pd.read_feather(in_path)

    columns = data.columns
    columns_cleaned = []
    columns_to_query = []
    for c in columns:
        if "." in c:
            c_cleaned = c.replace(".unit", "_unit")
            c_cleaned = c_cleaned[:c_cleaned.rfind(".")] + re.sub("\d", "", c_cleaned[c_cleaned.rfind(".") + 1:])
            c_cleaned = re.sub("[^\x00-\x7F]+", "", c_cleaned)
            columns_cleaned.append(c_cleaned)
            if "unit" not in c_cleaned:
                columns_to_query.append([c, c_cleaned])
        else:
            columns_cleaned.append(c)

    columns_df = pd.DataFrame(columns_to_query, columns = ["", "column_cleaned"]).set_index("")

    data_dictionary_df = pd.DataFrame([dataset_name] * len(columns), columns = ["Product Category"])
    data_dictionary_df.insert(1, "Column Name Raw", columns)
    data_dictionary_df.insert(2, "Column Name", columns_cleaned)
    data_dictionary_df.insert(3, "Column Top Values", [""]*len(columns))
    data_dictionary_df.insert(4, "Column Unit", [""]*len(columns))
    data_dictionary_df.insert(5, "Column Min", [""]*len(columns))
    data_dictionary_df.insert(6, "Column Max", [""]*len(columns))
    data_dictionary_df.insert(7, "Column Definition", [""]*len(columns))
    data_dictionary_df.insert(8, "Approved", [False]*len(columns))

    data_dictionary_df["Column Definition"][data_dictionary_df["Column Name"].str.contains("unit")] = "The unit of measure for the " + data_dictionary_df["Column Name"][data_dictionary_df["Column Name"].str.contains("unit")].str.replace("_unit", "") + " column."
    data_dictionary_df["Approved"][data_dictionary_df["Column Name"].str.contains("unit")] = True

    col_units = []
    col_values = []
    col_mins = []
    col_maxs = []
    for c in columns_df.index.values:
        try:
            col_unit = data[c + ".unit"].value_counts().nlargest(1).index[0]
        except:
            col_unit = "N/A"

        col_vals = str([str(val) for val in data[c].value_counts().nlargest(5).index.values]).replace("]", "").replace("[", "")

        if data[c].dtype == float or data[c].dtype == int:
            col_min = str(pd.DataFrame.min(data[c]))
            col_max = str(pd.DataFrame.max(data[c]))
        else:
            col_min = "N/A"
            col_max = "N/A"

        col_units.append(col_unit)
        col_values.append(col_vals)
        col_mins.append(col_min)
        col_maxs.append(col_max)

        data_dictionary_df["Column Unit"][data_dictionary_df["Column Name Raw"] == c] = col_unit
        data_dictionary_df["Column Top Values"][data_dictionary_df["Column Name Raw"] == c] = col_vals
        data_dictionary_df["Column Min"][data_dictionary_df["Column Name Raw"] == c] = col_min
        data_dictionary_df["Column Max"][data_dictionary_df["Column Name Raw"] == c] = col_max


    columns_df.insert(1, "column_values", col_values, True)
    columns_df.insert(2, "column_unit", col_units, True)
    columns_df.insert(3, "column_min", col_mins, True)
    columns_df.insert(4, "column_max", col_maxs, True)

    if not os.path.exists(out_dir + "/" + dataset_name):
        os.makedirs(out_dir + "/" + dataset_name)
        columns_df.to_csv(out_dir + "/" + dataset_name + "/columns_summary.csv", index = False)
        data_dictionary_df.to_csv(out_dir + "/" + dataset_name + "/" + dataset_name + "_Data_Dictionary.csv", index = False)
    else:
        columns_df.to_csv(out_dir + "/" + dataset_name + "/columns_summary.csv", index = False)
        data_dictionary_df.to_csv(out_dir + "/" + dataset_name + "/" + dataset_name + "_Data_Dictionary.csv", index = False)


@st.cache_data  
def query_LLM(column_summary: pd.DataFrame, column_name: str, dataset_description: str, LLM: str, LLM_token: str):

    if column_name not in column_summary["column_cleaned"].to_numpy():
        return -1

    values = column_summary[column_summary["column_cleaned"] == column_name]["column_values"].values[0]
    if pd.isna(column_summary[column_summary["column_cleaned"] == column_name]["column_unit"].values[0]): 
        unit = "\n"
    else: 
        unit = column_summary[column_summary["column_cleaned"] == column_name]["column_unit"].values[0]

    if pd.isna(column_summary[column_summary["column_cleaned"] == column_name]["column_min"].values[0]):
        range = "\n"
    else:
        min_val = column_summary[column_summary["column_cleaned"] == column_name]["column_min"].values[0]
        max_val = column_summary[column_summary["column_cleaned"] == column_name]["column_max"].values[0]
        range = "MIN VALUE: " + str(min_val) + " " + str(unit) + "\n\nMAX VALUE: " + str(max_val) + " " + str(unit) + "\n\n"

    prompt_params = {"dataset_description": dataset_description,
                "column_name": column_name,
                "values": values,
                "unit": unit,
                "range": range}

    PROMPT = f"""Given the following dataset file description, column name, and example values, please generate a column definition which is at least three sentences long and 100 characters in length. Generate this in dictionary format.

            DATASET FILE DESCRIPTION: {dataset_description}

            COLUMN NAME: {column_name}

            EXAMPLE VALUES: {values} {unit}
            {range}
            Given below is XML that describes the information to extract from this document and the tags to extract it into. 

            <output> 
                <string name="column_name" description="Name of the column in the data file"/>
                <string name="definition" format="min_length: min=100" description="Definition for the column"/> 
            </output> 

            ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. 

            Here are examples of simple (XML, JSON) pairs that show the expected behavior: 
            - `<string name='foo' format='two-words lower-case' />` => `{{{{'foo': 'example one'}}}}` 
            - `<list name='bar'><string format='upper-case' /></list>` => `{{{{"bar": ['STRING ONE', 'STRING TWO', etc.]}}}}`
            - `<object name='baz'><string name="foo" format="capitalize two-words" /><integer name="index" format="1-indexed" /></object>` => `{{{{'baz': {{{{'foo': 'Some String', 'index': 1}}}}}}}}`
            """

    if LLM == "Cohere":
        model = cohere.Client(api_key = LLM_token)
        try:
            output = model.generate(prompt = PROMPT.format(prompt_params), 
                                    model = 'command', 
                                    max_tokens = 1024, 
                                    temperature = 0.0)
            
            output = output.generations[0].text
            
            if "," in output[-3:]:
                output = output[:output.rfind(",")] + output[-1:]

            return output
        except Exception as e:
            return "Error in Cohere definition: {}".format(e)
    elif LLM == "LLaMA2":
        os.environ['REPLICATE_API_TOKEN'] = LLM_token

        query = replicate.run("meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                                input={"prompt": PROMPT.format(prompt_params)})
        
        output = ""
        for item in query:
            output = output + item

        try:
            return output
        except:
            return "Error"

@st.cache_data  
def query_LLM_TESTER(column_summary, column, dataset_description, LLM, LLM_key):
    time.sleep(1)
    return "Cache memory size in MB of the processor. Typical values range from 2.0 to 12.0, but values can be lower or higher."
    
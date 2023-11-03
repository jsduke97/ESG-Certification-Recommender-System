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

import altair as alt
import openai

from google.oauth2 import service_account
import gspread

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def log_response(current_log: pd.DataFrame, product_df: pd.DataFrame, mandate_df: pd.DataFrame, llm_prompt: str, llm_response_full: str, llm_response: str, LLM: str):
    # id | name | category_id | category_label | Sustainability certificates.42513 | 
    # Certification | Mandate Number | Mandate title | Mandate Description |
    # prompt | response | recommendation | model | rec_datetime
    
    id = product_df["id"].item()
    name = product_df["name"].item()
    category_id = product_df["category_id"].item()
    category_label = product_df["category_label"].item()
    certs = product_df["Sustainability certificates.42513"].item()

    cert = mandate_df["Certification"].item()
    mandate_no = mandate_df["Mandate Number"].item()
    mandate_title = mandate_df["Mandate title"].item()
    mandate_desc = mandate_df["Mandate Description"].item()

    current_log.loc[current_log.shape[0] + 1] = [id, name, category_id, category_label, certs, cert, mandate_no, mandate_title, 
                                                mandate_desc, llm_prompt, llm_response_full, llm_response, LLM, datetime.now()]

def find_last_filled_row(worksheet):
    return len(worksheet.get_all_values()) + 1

def save_recommendation(file_path: str, new_recommendation: pd.DataFrame):
    # id | name | category_id | category_label | Sustainability certificates.42513 | 
    # Certification | Mandate Number | Mandate title | Mandate Description |
    # prompt | response | recommendation | model | rec_datetime

    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    gc = gspread.authorize(credentials)

    sheet_url = st.secrets["spreadsheet1"]
    sheet = gc.open_by_url(sheet_url)

    conn = st.connection("gsheets", type=GSheetsConnection)

    worksheet = sheet.get_worksheet(0)  # Replace 0 with the index of your desired worksheet
    values = dataframe.values.tolist()

    # Find the last filled row
    last_filled_row = find_last_filled_row(worksheet)

    # Insert the data after the last filled row
    worksheet.insert_rows(values, last_filled_row)

    #df = pd.read_csv(file_path)
    #df = pd.concat([new_recommendation, df])
    #df.to_csv(file_path, index = False)

def prepare_mandate_query(mandate_df: pd.DataFrame, product: pd.DataFrame):
    # Take in the mandate information and columns relevant to the mandate, 
    # parse through the product data to ensure that accurate information 
    # will be sent to the LLM.

    # EXPECTS:
    # mandate_df: mandate definition and column mapping 
    #     columns: Certification, Mandate Number, Mandate title, Mandate Description, Column Name Raw, Column Name, Column Rank
    
    # product: product dataset
    #     columns: name, and column names corresponding to Column Name Raw from mandate_df

    mandate_header = "MANDATE\n\n{} Certification\nMandate {}: {}".format(mandate_df.iloc[0]["Certification"], mandate_df.iloc[0]["Mandate Number"], mandate_df.iloc[0]["Mandate title"])
    mandate_description = "\nMandate Description: \n{}".format(mandate_df.iloc[0]["Mandate Description"])

    product_name = "\n\nPRODUCT\n\nName: {} ({})".format(product.iloc[0]["name"], product.iloc[0]["category_label"])

    product_attributes = 0
    product_attribute_string = "\n"
    i = mandate_df.index[0]
    while product_attributes < 5 and i < mandate_df.index[-1]:
        col = mandate_df["Column Name Raw"][i]
        if not pd.isna(product[col][0]):
            col_unit = product[col + ".unit"][0]
            if pd.isna(col_unit):
                col_unit = ""
            else:
                col_unit = " " + str(col_unit)
            product_attribute_string += str(mandate_df["Column Name"][i]) + ": " + str(product[col][0]) + str(col_unit) + "\n"
            product_attributes += 1
        i += 1

    final_query = "\nIs the product \"{}\" compliant with the {} Certification Mandate {}: {}?".format(product.iloc[0]["name"], 
                                                                                                            mandate_df.iloc[0]["Certification"], 
                                                                                                            mandate_df.iloc[0]["Mandate Number"], 
                                                                                                            mandate_df.iloc[0]["Mandate title"])

    return mandate_header + mandate_description + product_name + product_attribute_string + final_query

@st.cache_data  
def query_LLM(mandate_df: pd.DataFrame, mandate_column_df: pd.DataFrame, product: pd.DataFrame, LLM: str, LLM_token: str):

    payload = prepare_mandate_query(mandate_column_df, product)

    prompt = """You are a subject matter expert for assessing the eligibility of IT products for sustainability certifications. 
    Given the following certification mandate, assess whether the product meets the mandate using the product attributes. Provide 
    the following as your assessment:

    - Recommendation: TRUE if the product is compliant, FALSE if the product is not compliant or if more information is needed

    - Reasoning: Your reasoning for the recommendation.

    {}

    If there is not enough information provided, respond with "MORE INFO NEEDED" and your reasoning. 
    """.format(payload)

    #st.markdown(prompt)

    if LLM == "Cohere":
        model = cohere.Client(api_key = LLM_token)
        try:
            output = model.generate(prompt = prompt, 
                                    model = 'command', 
                                    max_tokens = 1024, 
                                    temperature = 0.0)
            
            output = output.generations[0].text
            
            if "," in output[-3:]:
                output = output[:output.rfind(",")] + output[-1:]

            #st.markdown(output)

            return prompt, output
        except Exception as e:
            if "You are using a Trial key" in str(e):
                return prompt, "LIMIT RATE"
            else:
                return prompt, "Error in Cohere response: {}".format(e)
    elif LLM == "LLaMA2":
        os.environ['REPLICATE_API_TOKEN'] = LLM_token

        query = replicate.run("meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                                input={"prompt": prompt})
        
        output = ""
        for item in query:
            output = output + item

        try:
            return prompt, output
        except:
            return prompt, "Error"
    elif LLM == "GPT-3.5":
        openai.api_key = LLM_token

        try: 
            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            if "ServiceUnavailableError" in str(e):
                return prompt, "ServiceError"
            else:
                return prompt, "Error in OpenAI Response:{}".format(e)


        return prompt, output['choices'][0]['message']['content']

@st.cache_data  
def query_LLM_TESTER(mandate_df: pd.DataFrame, product: pd.DataFrame, LLM: str, LLM_token: str):
    time.sleep(.05)

    return (np.random.rand(1) > .3)[0]


def output_responses(output_df: pd.DataFrame, cert: str, LLM: str):
    df = output_df[output_df["Certification"] == cert][output_df["model"] == LLM]
    mandates_passed = df[df["recommendation"] == "True"].shape[0]
    mandates_failed = df[df["recommendation"] == "False"].shape[0]
    mandates_na = df[df["recommendation"] == "N/A"].shape[0]
    mandates_total = len(df["recommendation"])
    try:
        percent_passsed = round(mandates_passed / (mandates_passed + mandates_failed) * 100)
    except:
        percent_passsed = 100
    st.markdown("{} assessement:".format(LLM))

    t1 = [cert, "Passed", mandates_passed, 1, "green"]
    t2 = [cert, "Failed", mandates_failed, 2, "red"]
    t3 = [cert, "More Info Needed", mandates_na, 3, "yellow"]

    chart_data = pd.DataFrame([t1, t2, t3], columns=["Cert", "Status", "Count", "Order", "Color"])

    c = (
    alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('sum(Count)', axis = None, sort = ["Passed", "Failed", "N/A"]),
        y=alt.Y('Cert', axis = None),
        color=alt.Color('Color',
            legend=None, 
            scale = alt.Scale(domain=['green', 'red', 'yellow'], range=['mediumspringgreen', 'palevioletred', 'lightyellow']),),
        order=alt.Order(
        # Sort the segments of the bars by this field
        'Order',
        sort='ascending'
        ),
        tooltip=['Status', 'Count']
    ))

    st.altair_chart(c, use_container_width=True)
    return mandates_passed, mandates_failed, mandates_na, percent_passsed
    
    
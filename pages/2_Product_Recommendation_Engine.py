import streamlit as st 
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode # pip install streamlit-aggrid
import os
import numpy as np
import cert_eval_functions as cef
from streamlit_js_eval import streamlit_js_eval
import json
import time
import altair as alt

st.markdown('# Product Recommendation Engine')
st.sidebar.markdown('Product Certification Evaluator')

st.cache_data.clear()

if 'page' not in st.session_state:
    st.session_state.page = None
if 'state' not in st.session_state:
    st.session_state.state = None
if 'certs' not in st.session_state:
    st.session_state.certs = []
if 'models' not in st.session_state:
    st.session_state.models = []
if 'product' not in st.session_state:
    st.session_state.product = []
if 'master_path' not in st.session_state:
    st.session_state.master_path = ""
if 'summary_path' not in st.session_state:
    st.session_state.summary_path = ""
if 'rec' not in st.session_state:
    st.session_state.rec = None

def set_page(page):
    st.session_state.page = page

def set_page_save(page, detail_df, file_path):
    cef.save_recommendation(file_path, detail_df, "full")
    st.session_state.page = None
    st.session_state.product = []
    st.session_state.rec = None

# list of folders in ./Datasets
datasets = np.sort(next(os.walk("./Datasets"))[1])

# master csv of files
master_file_list_path = "./file_list.csv"

# Create buttons 
selected_dataset = st.selectbox("Select Product Dataset", datasets, index=0, placeholder="Select dataset")

# master csv of recs
st.session_state.master_path = "./Product Certification/" + selected_dataset + "/product_mandate_recommendation.csv"
st.session_state.summary_path = "./Product Certification/" + selected_dataset + "/product_recommendation_summary.csv"
rec_file_columns = pd.read_csv(st.session_state.master_path).columns
st.session_state.rec = pd.DataFrame([], columns = rec_file_columns)

cohere_key = st.text_input("Cohere API Key", type = "password")
replicate_key = st.text_input("Replicate API Key", type = "password")
openai_key = st.text_input("OpenAI API Key", type = "password")

api_keys = {"Cohere": cohere_key, "LLaMA2": replicate_key, "GPT-3.5": openai_key}

st.divider()

st.session_state.certs = st.multiselect("ESG Certifications (select all that apply)", ["TCO", "Energy Star"], default = ["TCO", "Energy Star"]) # User can select both

st.session_state.models = st.multiselect("LLM model (select all that apply)", ["Cohere", "LLaMA2", "GPT-3.5"], default = ["Cohere"]) # User can select both

st.divider()

st.markdown("Search for a product by name and generate a recommendation for the selected ESG certifications.")

products_df = pd.read_csv(master_file_list_path)
products_df = pd.read_csv(products_df[products_df["file_folder"] == selected_dataset].head(1)["file_name"][0])

mandates_df = pd.read_csv("./Product Certification/certification_mandates_revised.csv")
mandate_column_full_df = pd.read_csv("./Product Certification/" + selected_dataset + "/mandate_column_relevance_full.csv")

st.session_state.product = st.multiselect("Product Search:", products_df["name"], max_selections= 1) # User can search for product

if st.session_state.product != []:
    product_df = products_df[products_df["name"] == st.session_state.product[0]]

if st.session_state.product != []:
    st.button("Generate Recommendation", use_container_width = True, on_click=set_page, args=["Generate New"])

if st.session_state.page == "Generate New":

    try: 
        st.markdown("<h5 style= 'text-align: center;'>Product Name: " + st.session_state.product[0] + "</h5>", unsafe_allow_html= True)
    except:
        st.markdown("")

    for cert in st.session_state.certs:

        output_data = []
        
        cert_cols = st.columns(len(st.session_state.models) + 1)
        cert_cols_i = 1

        mandates_cert = mandates_df[mandates_df["Certification"] == cert]

        for LLM in st.session_state.models:
            progress_text = "Querying {} for {} recommendation...{} out of {} complete."
            cert_progress = st.progress(0, text=progress_text.format(LLM, cert, 0, mandates_cert.shape[0]))
            start_time = time.time()
            for mandate in range(1):#mandates_cert.shape[0]):

                mandate_df = mandates_cert.iloc[[mandate]]
                mandate_column_df = mandate_column_full_df[mandate_column_full_df["Certification"] == cert]
                mandate_column_df = mandate_column_df[mandate_column_df["Mandate Number"] == mandate_df["Mandate Number"].item()]

                # st.dataframe(product_df)
                #st.dataframe(mandate_df)
                #st.dataframe(mandate_column_df)

                prompt, llm_response_full = cef.query_LLM(mandate_df, mandate_column_df, product_df, LLM, api_keys[LLM])

                if llm_response_full == "LIMIT RATE":
                    st.cache_data.clear()
                    time.sleep(61)
                    prompt, llm_response_full = cef.query_LLM(mandate_df, mandate_column_df, product_df, LLM, api_keys[LLM])
                    #st.markdown(llm_response_full)

                if llm_response_full == "ServiceUnavailableError":
                    st.cache_data.clear()
                    time.sleep(10)
                    prompt, llm_response_full = cef.query_LLM(mandate_df, mandate_column_df, product_df, LLM, api_keys[LLM])

                cert_progress.progress(((mandate + 1) / mandates_cert.shape[0]), text=progress_text.format(LLM, cert, (mandate + 1), mandates_cert.shape[0]))
                
                if "TRUE" in llm_response_full or "does meet" in llm_response_full:
                    llm_response = "True"
                elif "more info" in llm_response_full.lower() or "not provided" in llm_response_full.lower() or "cannot determine" in llm_response_full.lower():
                    llm_response = "N/A"
                else: 
                    llm_response = "False"

                #st.markdown(llm_response)

                cef.log_response(st.session_state.rec, product_df, mandate_df, prompt, llm_response_full, llm_response, LLM)
                #st.dataframe(st.session_state.rec)
            cert_progress.empty()
            
            elapsed_time = time.time() - start_time

            with cert_cols[cert_cols_i]:
                output = st.session_state.rec[st.session_state.rec["Certification"] == cert][st.session_state.rec["model"] == LLM]
                passed, failed, na, rec_per = cef.output_responses(output, cert, LLM)
                summary_df = pd.DataFrame([[st.session_state.product[0], LLM, cert, passed, failed, na, rec_per, round(elapsed_time), 0]],
                            columns = ["product", "model", "cert", "mandates passed", "mandates failed", "mandates na", "percentage_passed", "time", "cost"])

                cef.save_recommendation(st.session_state.summary_path, summary_df, "summary")
                output_data.append(rec_per)
                cert_cols_i += 1

        with cert_cols[0]:
            if np.min(output_data) >= 55:
                text_color = "green"
                recommendation = "##### Good Candidate"
            else:
                text_color = "red"
                recommendation = "##### Not a Good Candidate"
            st.markdown("##### :{}[{}]".format(text_color, cert)) 
            st.markdown("##### :{}[{}%]".format(text_color, np.min(output_data))) 
            st.markdown(recommendation)

    st.markdown("Details:")
    st.dataframe(st.session_state.rec)

    st.button("Export", 
            use_container_width = True, 
            on_click=set_page_save, 
            args=["Export", st.session_state.rec, st.session_state.master_path])
import streamlit as st 
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode # pip install streamlit-aggrid
import os
import numpy as np
import data_dictionary_functions as ddf
from streamlit_js_eval import streamlit_js_eval
import json
import time
from google.oauth2 import service_account
import gspread

st.markdown('# Data Dictionary Creator')
st.sidebar.markdown('Data Dictionary Creator')
demo_toggle = st.sidebar.toggle("Demo Mode", True)

if 'page' not in st.session_state:
    st.session_state.page = None
if 'selected_rows' not in st.session_state: 
    st.session_state.selected_rows = []
if 'state' not in st.session_state:
    st.session_state.state = None
if 'definition' not in st.session_state:
    st.session_state.definition = None

def set_page(page):
    st.session_state.page = page

# list of folders in ./Datasets
datasets = np.sort(next(os.walk("./Datasets"))[1])

# master csv of files
master_file_list_path = "./file_list.csv"

# Create buttons 
selected_dataset = st.selectbox("Select Product Dataset", datasets, index=0, placeholder="Select dataset")

st.button("Upload New Dataset", on_click=set_page, args=["Upload"])

cohere_key = st.text_input("Cohere API Key", type = "password")
replicate_key = st.text_input("Replicate API Key", type = "password")

api_keys = {"Cohere": cohere_key, "LLaMA2": replicate_key}

st.divider()

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
gc = gspread.authorize(credentials)

col1, col2 = st.columns(2)
with col1:
    st.button("Review Data Definitions", use_container_width = True, on_click=set_page, args=["Review"])

with col2:
    st.button("Create New Definitions", use_container_width = True, on_click=set_page, args=["Create"])

# Data Dictionary Creator: If user clicks on Upload New File link 
if st.session_state.page == "Upload":
    uploaded_file =  st.file_uploader("Please click here to upload a csv or feather file"
                                     ,type = ['csv','feather']) # If we want to accept multiple files, accept_multiple_files=True
    if uploaded_file is not None: 
       new_dataset_name = st.text_input("Dataset Name", value = "")
       new_dataset_description = st.text_area("Dataset Description", 
                                              value = "A short description of the products and features in the dataset")
       
       file_name = uploaded_file.name
       # file_names.append(file_name)
       file_type = file_name[file_name.rfind("."):]
       file_folder = "./Datasets/" + new_dataset_name
       file_path = file_folder + "/" + file_name
       
    if st.button("Submit"):
        if not os.path.exists(file_folder):
            st.session_state.state = 1
        else:
            st.session_state.state = 2

if st.session_state.state == 1:
    os.makedirs(file_folder)
    pbar = st.progress(0, text="Submitting...")

    if file_type == ".csv":
        pd.read_csv(uploaded_file).to_csv(file_path, index=False)
    elif file_type == ".feather":
        pd.read_feather(uploaded_file).to_csv(file_path, index=False)

    pbar.progress(40, text="Submitting...")

    ddf.add_file_to_master(master_file_list_path, new_dataset_name, file_path, new_dataset_description)
    pbar.progress(60, text="Submitting...")
    ddf.create_column_summary(file_path, file_type, "./Data Dictionary Output", new_dataset_name)
    pbar.progress(100, text="Done")

    streamlit_js_eval(js_expressions="parent.window.location.reload()")

if st.session_state.state == 2:
    st.write("A dataset exists with this name, please choose a new name or overwrite.")
    if st.button("Overwrite Dataset"):
        pbar = st.progress(0, text="Submitting...")

        for filename in os.listdir("./Datasets/" + new_dataset_name):
            file_path = os.path.join("./Datasets/" + new_dataset_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        pbar.progress(10, text="Submitting...")

        if file_type == ".csv":
            pd.read_csv(uploaded_file).to_csv(file_path, index=False)
        elif file_type == ".feather":
            pd.read_feather(uploaded_file).to_csv(file_path, index=False)

        pbar.progress(40, text="Submitting...")

        ddf.add_file_to_master(master_file_list_path, new_dataset_name, file_path, new_dataset_description)
        pbar.progress(60, text="Submitting...")
        ddf.create_column_summary(file_path, file_type, "./Data Dictionary Output", new_dataset_name)
        pbar.progress(100, text="Done.")
        
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    

# Data Dictionary Creator: If user clicks on Review Data Definitions link
# We can create a Data Dictionary Class and import it to process the csv file and call LLM for definitions 
if st.session_state.page == "Review":
    st.header("Column Name Definitions for Notebooks Dataset")
    st.write("Select all rows for which you want to change the definition")
    # Example: Notebooks CSV storing the already cleaned column definitions 
    if demo_toggle:
        sheet_url = st.secrets["DD_spreadsheet"]
        sheet = gc.open_by_url(sheet_url).get_worksheet(0)
        df_column_desc = pd.DataFrame(sheet.get_all_values())
        df_column_desc.columns = df_column_desc.iloc[0]
        df_column_desc.drop(df_column_desc.index[0], inplace = True)
        df_column_desc = df_column_desc[6:]
        df_column_desc = df_column_desc[~df_column_desc["Column Name"].str.contains("unit")]
    else:
        path = './Data Dictionary Output/'+ selected_dataset + '/' + selected_dataset + '_Data_Dictionary.csv'
        df_column_desc = pd.read_csv(path) 
    # Display dataset allowing users to navigate and click on desired column names they wish to change
    gd = GridOptionsBuilder.from_dataframe(df_column_desc[['Column Name','Column Definition']])
    gd.configure_pagination(paginationAutoPageSize=False,paginationPageSize = 10)
    gd.configure_selection(selection_mode= 'multiple', use_checkbox = True)
    gd.configure_grid_options(alwaysShowHorizontalScroll=True, enableRangeSelection=True, pagination=True, paginationPageSize=10000, domLayout='normal')
    gd.configure_column("Column Definition", singleClickEdit = True, wrapText = True, width = 500, autoHeight = True)
    gridoptions = gd.build()
    table = AgGrid(df_column_desc[['Column Name','Column Definition']], gridOptions = gridoptions, 
                            update_mode = GridUpdateMode.SELECTION_CHANGED, height=500)
    # Store the selected rows 
    try:
        st.session_state.selected_rows = df_column_desc.index[df_column_desc["Column Name"].isin(pd.DataFrame(table['selected_rows'])["Column Name"])].tolist()
    except:
        st.session_state.selected_rows = []

    if st.button("Continue"):
        #df_column_desc["Approved"] = ~df_column_desc["Column Name"].isin(st.session_state.selected_rows)
        if demo_toggle:
            for column_i in st.session_state.selected_rows:
                sheet.update_cell(column_i + 1, 9, False)
        
        st.session_state.page = "Create"
        st.rerun()



# Data Dictionary Creator: If user clicks on Create New Definitions link
if st.session_state.page == "Create":
    column_summary = pd.read_csv("./Data Dictionary Output/" + selected_dataset + "/columns_summary.csv")
    
    dataset_description = pd.read_csv(master_file_list_path)
    dataset_description = dataset_description[dataset_description["file_folder"] == selected_dataset].head(1)["file_description"][0]

    if demo_toggle == True:
        sheet_url = st.secrets["DD_spreadsheet"]
        sheet = gc.open_by_url(sheet_url).get_worksheet(0)
        data_dictionary = pd.DataFrame(sheet.get_all_values())
        data_dictionary.columns = data_dictionary.iloc[0]
        data_dictionary.drop(data_dictionary.index[0], inplace = True)
        data_dictionary = data_dictionary[6:]
        data_dictionary = data_dictionary[~data_dictionary["Column Name"].str.contains("unit")]
    else:
        data_dictionary = pd.read_csv("./Data Dictionary Output/" + selected_dataset + "/" + selected_dataset + "_Data_Dictionary.csv")
    unapproved_columns = data_dictionary[data_dictionary["Approved"] == "FALSE"]
    
    selected_model = st.multiselect("LLM model (select all that apply)", ["Cohere", "LLaMA2"], default = ["Cohere", "LLaMA2"]) # User can select both

    st.markdown("<p style= 'text-align: center;'>1 of " + str(len(unapproved_columns)) + "</p>", unsafe_allow_html= True)

    # Grab next unapproved column
    column = unapproved_columns.iloc[0]["Column Name"]
    st.markdown("Column Name: :green[{}]".format(column))

    col_sum1, col_sum2 = st.columns(2)

    values = column_summary[column_summary["column_cleaned"] == column]["column_values"].values[0]
    with col_sum1:
        st.markdown("Most Common Values: :green[{}]".format(values))
        if pd.isna(column_summary[column_summary["column_cleaned"] == column]["column_min"].values[0]):
            range = "\n"
            st.markdown("Column Minimum: :green[N/A]")
        else:
            min_val = column_summary[column_summary["column_cleaned"] == column]["column_min"].values[0]
            st.markdown("Column Minimum: :green[{}]".format(min_val))
    
    with col_sum2:
        if pd.isna(column_summary[column_summary["column_cleaned"] == column]["column_unit"].values[0]): 
            unit = "\n"
            st.markdown("Column Unit: :green[N/A]")
        else: 
            unit = column_summary[column_summary["column_cleaned"] == column]["column_unit"].values[0]
            st.markdown("Column Unit: :green[{}]".format(unit))

        if pd.isna(column_summary[column_summary["column_cleaned"] == column]["column_min"].values[0]):
            range = "\n"
            st.markdown("Column Maximum: :green[N/A]")
        else:
            max_val = column_summary[column_summary["column_cleaned"] == column]["column_max"].values[0]
            st.markdown("Column Maximum: :green[{}]".format(max_val))


    definitions = {}

    user_1, user_2 = st.columns([2,8])
    with user_1: 
        st.markdown("User/Existing Definition")
    
    existing_definition = data_dictionary["Column Definition"][data_dictionary["Column Name"]==column].iloc[0]
    if pd.isna(existing_definition):
        existing_definition = ""

    with user_2:
        user_def_final = st.text_area("", existing_definition, label_visibility="collapsed",  height = 100)
        definitions["User"] = user_def_final
    
    if column not in column_summary["column_cleaned"].to_numpy():
        st.write("ID Column, Please enter Definition manually or accept 'ID Column for the file. Definition not applicable.'")

    # add LLM results if selected
    if "Cohere" in selected_model:
        co_1, co_2 = st.columns([2,8])
        with co_1:
            st.markdown("Cohere Definition")

        
        co_definition = ddf.query_LLM(column_summary, column, dataset_description, "Cohere", api_keys["Cohere"], demo_toggle)

        if co_definition == -1:
            co_definition = 'ID Column for the file. Definition not applicable.'
        else:
            try:
                co_definition = json.loads(co_definition)["definition"]
            except:
                try:
                    co_definition = list(json.loads(co_definition).values())[0]
                except:
                    try: 
                        co_definition = co_definition["definition"]
                    except:
                        st.write(co_definition)
                        co_definition = "Error"

        with co_2:
            co_def_final = st.text_area("", co_definition, label_visibility="collapsed", height = 120)
            definitions["Cohere"] = co_def_final

    if "LLaMA2" in selected_model:
        la_1, la_2 = st.columns([2,8])
        with la_1:
            st.markdown("LLaMA2 Definition")

        
        la_definition = ddf.query_LLM(column_summary, column, dataset_description, "LLaMA2", api_keys["LLaMA2"], demo_toggle)

        try:
            la_definition = json.loads(la_definition)["definition"]
        except:
            try: 
                la_definition = la_definition["definition"]
            except:
                la_definition = "Error"

        if la_definition == -1:
            la_definition = 'ID Column for the file. Definition not applicable.'

        with la_2:
            la_def_final = st.text_area("", la_definition, label_visibility="collapsed", height = 120, key = "llama_def")
            definitions["LLaMA2"] = la_def_final

    st.write("APPROVE DEFINITION:")

    appr_1, appr_2 = st.columns([3,1])

    with appr_1:
        definition_selection = st.radio("Approve Definition", ["User", "Cohere", "LLaMA2"], horizontal = True, 
                                        label_visibility="collapsed")

    with appr_2:
        if st.button("Approve", "approve_definition", use_container_width=True):
            if demo_toggle:
                column_i = data_dictionary.index[data_dictionary["Column Name"] == column].item()
                sheet.update_cell(column_i + 1, 8, definitions[definition_selection])
                sheet.update_cell(column_i + 1, 9, True)
            else:
                data_dictionary["Column Definition"][data_dictionary["Column Name"] == column] = definitions[definition_selection]
                data_dictionary["Approved"][data_dictionary["Column Name"] == column] = True 
                data_dictionary.to_csv("./Data Dictionary Output/" + selected_dataset + "/" + selected_dataset + "_Data_Dictionary.csv", index = False)
            st.session_state.page = "Create"
            st.rerun()

    if st.button("Rerun Query"):
        st.cache_data.clear()


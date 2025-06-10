import os
import pandas as pd
from datetime import date
import gradio as gr
from pyvis.network import Network
import ast
from openai import OpenAI
import json 
import string
from datetime import datetime
import random
from utils import get_geometries
from scenario_generation import generate_new_relations
import geopandas as gpd
import folium
from shapely.geometry import mapping
import dropbox
from dropbox.exceptions import ApiError
import io 

# Dropbox access token and file path
ACCESS_TOKEN = 'sl.u.AFyJ21_uWvAAQlqIm8tZVU7ANrri9q8lT-wQAPYtuAMbRXSmT_SDDC0c_S6-MYDMvDsr88CcCsiFT1gjqFeh1-ndfTHrZD7JNQ4Quy_UFDOjwYd21ESyLLgSJvzPA7wDc2x4rAFS-oMerYqap4x5CxtFzg9UNRePrw-TFBmylOgk8UFBXeTjMy9RXBjILfjSqBrV7CUf-hA14QqxE0bU40iKtym8uCAvoZvWJy2gfj0lu8d7wiH4Ou9giXG8InHotOlOSImjUO9bMJJlIvTG3G1H_bWqPGPLnwOT80dtNvns3TTqbSSw8ZJREvo_YtciHQajOmShjKJ8V-0MNgNrdr1QEZ0HmQP-kaZu9bFqxqo8gblLc4JcFIbP8-axDDU4aZvzARDOEM3ibK1Ve3D_Fy7T66GN4oE9MUr6YT1NC9lANs3yOPjvK5M8bFTLa6ArX-AygFkkV8iW8BZkeDhAGv7THAjHp1Sk4j8xM0dUR9sHTwgE36wK0_kD5F2IucqM9ljCU2CpZGmWOjX4AzDzhRZwccGc-g1RibFibMonnrQXhYdmIKPJW372oQBJikyeuBQEAKMsGRRNQ5eHFBJEWYqOZdsjEYpfiu8be-scvwcYazho0-DHU7rychWhbYR4iKATHnCsLbnBAe6ELciaivSL0NCzdIhTgb-YJRhGDVWZoFNtqHIvMLP0wuVwU2kafA8s7G-ZNHWn0qpjTwB59HXk3OLJoKMK-Utk_nuQ0dx4VzlUTug-yC7VHtrB5qcWhW4pCuEYH5gN9AXf42acxqeRMxpgriVeDlEv7Fzn4ni5WHwTvr1YnQH0yv-S9dv0F1gx6f4gOE7dFTGY9HmW6oq0m4E62BhwKEeGFB5mPVX0usq1uuNdRKdrHYsVgomqKSWdrbRIxR1RlsRBx1vaaTVvxI1JOjxaWyFxDTC3-m6LCfnL-4Y1V5oXIK22CLBE7Zs49DJAou9K6asLtOsHWqErjtbEzKmTwU_xm8S0cMjhm31UYvtn83N8j9Qxkmo70a-wPAn5i_pU54Cb3aQKp9K6rLaKojILN3NWvitixLBGVgWNRcQYc4S11WgkoXABobK0fMsoNjvqo4BDfrj_nqJ2PbgO5ElyVpPtRBkd6jiil0DQvqocFjDmqP-Ce2Bql-gp0ndK8rdLhuX9XJ0e7TrAWiJXZLvj7hBKodCcGdyKCux40Cum6kvm_58Jfoic6TXQ5t1O8kiprh1oSvZys2enmp2ZNSiOz9rwQZabLS7--aa62ODWb66c0FdKhG66pnY0PvnseOkdLqVy19IXH-MXwiGYayFNaW_YKD_qdYJPfJov0dTYKliIn_0wnDZen4Dewk4QRpWgKwEphBsmUGEcipM5Wk9YjREqA9CTKwWdS09kgYFhM2jr2jml9_40aMWgj3EGrdT1GGhxR-XJG7Tt0KHmjG4HP7WhW84dQADyVOe-0kSmbpDgCbrPN7ac9XA'

# Load the Natural Earth dataset from the extracted shapefile
world = gpd.read_file('./data/ne_110m_admin_0_countries.shp')

EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    EMM_RETRIEVERS_OPENAI_API_KEY = config['EMM_RETRIEVERS_OPENAI_API_KEY']
#EMM_RETRIEVERS_OPENAI_API_KEY = os.environ['key_gptjrc']
    
client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat2.csv", sep=',', header=0, dtype=str, encoding='utf-8')

# Function to get fallback coordinates from GeoPandas
def get_country_centroid(country_name):
    # Filter the world GeoDataFrame for the country
    country = world[world['NAME'] == country_name]
    if not country.empty:
        # Get the centroid of the country's geometry
        centroid = country.geometry.centroid.iloc[0]
        return (centroid.y, centroid.x)
    else:
        # Default to (0, 0) if country not found
        return (0, 0)

# Function to plot a geometry using Folium
def plot_geometry_folium(geometry, location_name='Location', country_name=None):
    if geometry is not None:
        # Get the centroid for initial map location
        centroid = geometry.centroid
        initial_coords = (centroid.y, centroid.x)
    else:
        # Use geopandas to get fallback coordinates
        initial_coords = get_country_centroid(country_name)

    # Create the map centered at initial_coords
    m = folium.Map(location=initial_coords, zoom_start=6)

    if geometry is not None:
        # Convert to GeoJSON for Folium if geometry exists
        geo_json = mapping(geometry)
        # Add GeoJSON to the map
        folium.GeoJson(geo_json, name=location_name).add_to(m)
    else:
        # Add a marker to indicate the country location
        folium.Marker(initial_coords, popup=location_name).add_to(m)

    # Return the HTML representation of the map object
    return m._repr_html_()


def gpt_story(storyline):
    prompt = (
        "Use the information provided to create a short, clear, and useful narrative about a disaster event. "
        "The goal is to help decision-makers (e.g. policy makers, disaster managers, civil protection) understand what happened, why, and what it caused. "
        "Keep it short and focused.\n\n"
        "Include all key information, but keep the text concise and easy to read. Avoid technical jargon.\n\n"
        "Steps to Follow:\n"
        "1. Start with what happened: Briefly describe the disaster event (what, where, when, who was affected).\n"
        "2. Explain why it happened: Use the evidence provided to describe possible causes or triggers (e.g. heavy rainfall, poor infrastructure, heatwave).\n"
        "3. Show the impacts: Highlight key impacts such as fatalities, displacement, health effects, or damage.\n"
        "4. Connect the dots: Show how different factors are linked. Use simple cause-effect language (e.g. drought led to crop failure, which caused food insecurity).\n"
        "5. Mention complexity if needed: If there were multiple contributing factors or reinforcing effects (e.g. climate + conflict), briefly explain them.\n"
        "6. Keep it useful: Write with a decision-maker in mind. Focus on what matters: drivers, impacts, and lessons for preparedness or response.\n\n"
        f"Information: {storyline}"
    )

    completion = client1.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are a disaster manager expert in risk dynamics."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    message_content = completion.choices[0].message.content
    return message_content


# DataFrame to store evaluation data
evaluation_df = pd.DataFrame(columns=["DisNo.", "TPN", "TPL", "FPN", "FPL", "FNN", "FNL", "User ID"])



def try_parse_date(y, m, d):
    try:
        if not y or not m or not d:
            return None
        return date(int(float(y)), int(float(m)), int(float(d)))
    except (ValueError, TypeError):
        return None

def plot_cgraph_pyvis(grp):
    if not grp:
        return "<div>No data available to plot.</div>"

    net = Network(notebook=False, directed=True)
    edge_colors_dict = {"causes": "red", "prevents": "green"}

    for src, rel, tgt in grp:
        src = str(src)
        tgt = str(tgt)
        rel = str(rel)
        net.add_node(src, shape="circle", label=src)
        net.add_node(tgt, shape="circle", label=tgt)
        edge_color = edge_colors_dict.get(rel, 'black')
        net.add_edge(src, tgt, title=rel, label=rel, color=edge_color)

    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')

    html = net.generate_html()
    html = html.replace("'", "\"")

    # Adjust the iframe style to center the graph and fit the container
    html_s = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <iframe style="width: 90%; height: 800px; margin: 0 auto;" name="result" allow="midi; geolocation; microphone; camera; 
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
        allow-scripts allow-same-origin allow-popups 
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>
    </div>
    """
    
    return html_s

def generate_unique_user_id():
    # Generate a timestamp string
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate a random string of 5 letters
    random_str = ''.join(random.choices(string.ascii_letters, k=5))
    # Combine both to form a unique User ID
    return f"{timestamp_str}_{random_str}"

def load_initial_data():
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    try:
        # Try to download the existing CSV file from Dropbox
        metadata, res = dbx.files_download(DROPBOX_FILE_PATH)
        csv_content = res.content.decode('utf-8')
        # Read the CSV content into a DataFrame
        df = pd.read_csv(io.StringIO(csv_content))
        print("Loaded existing data from Dropbox.")
    except ApiError as e:
        # If file not found, initialize an empty DataFrame
        if e.error.is_path() and e.error.get_path().is_not_found():
            df = pd.DataFrame(columns=["DisNo.", "User Feedback", "User ID"])
            print("No existing file found on Dropbox. Initialized an empty DataFrame.")
        else:
            print(f"Error downloading file: {e}")
            df = pd.DataFrame(columns=["DisNo.", "User Feedback", "User ID"])
    return df

def append_to_csv_on_dropbox(new_content, dropbox_path):
    dbx = dropbox.Dropbox(ACCESS_TOKEN)

    try:
        # Try to download existing file content
        metadata, res = dbx.files_download(dropbox_path)
        existing_content = res.content.decode('utf-8')
    except ApiError as e:
        # If file not found, start with empty content
        if e.error.is_path() and e.error.get_path().is_not_found():
            existing_content = ''
        else:
            print(f"Error downloading file: {e}")
            return

    # Append new content without header if file already exists
    if existing_content:
        new_content_lines = new_content.splitlines()
        new_content_without_header = '\n'.join(new_content_lines[1:])
        combined_content = existing_content.rstrip('\n') + '\n' + new_content_without_header
    else:
        combined_content = new_content

    try:
        # Upload combined content back to Dropbox (overwrite)
        dbx.files_upload(combined_content.encode('utf-8'), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
        print(f"Appended and uploaded to {dropbox_path} successfully!")
    except Exception as e:
        print(f"Error uploading file: {e}")

def save_data_to_dropbox():
    # Convert DataFrame to CSV string
    csv_content = evaluation_df.to_csv(index=False)
    # Append the CSV content to the file on Dropbox
    append_to_csv_on_dropbox(csv_content, DROPBOX_FILE_PATH)

def save_data(dis_no, user_feedback):
    global evaluation_df

    if not dis_no or dis_no == "Select a Disaster Event":
        print("Invalid input. Ensure a disaster event is selected.")
        return

    user_id = generate_unique_user_id()
    new_data = pd.DataFrame([[dis_no, user_feedback, user_id]], 
                            columns=["DisNo.", "User Feedback", "User ID"])
    evaluation_df = pd.concat([evaluation_df, new_data], ignore_index=True)
    print("Updated DataFrame:")
    print(evaluation_df)

    save_data_to_dropbox()
    print(f"Data saved: DisNo: {dis_no}, Feedback: {user_feedback}, User ID: {user_id}")


DROPBOX_FILE_PATH = '/evaluation_data.csv'
evaluation_df = load_initial_data()


def update_row_dropdown(disaster_type=None, country=None):
    # Start with the entire dataframe
    filtered_df = df

    # Step 1: Filter by Disaster Type
    if disaster_type:
        filtered_df = filtered_df[filtered_df['Disaster Type'] == disaster_type]

    # Step 2: Further filter by Country
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    # Step 3: Generate and sort the DisNo. choices based on the filtered DataFrame
    choices = sorted(filtered_df['DisNo.'].tolist()) if not filtered_df.empty else []

    # Add a placeholder option at the beginning
    choices = ["Select a Disaster Event"] + choices

    print(f"Available DisNo. for {disaster_type} in {country}: {choices}")
    
    # Return the update for the dropdown, defaulting to the placeholder
    return gr.update(choices=choices, value=choices[0] if choices else None)



def display_info(selected_row_str, country):
    if not selected_row_str or selected_row_str == 'Select a Disaster Event':
        print("No valid disaster event selected.")
        return ('No valid event selected.', '<div>No graph available.</div>', '', '', '')

    print(f"Selected Country: {country}, Selected Row: {selected_row_str}")

    # Filter the dataframe for the selected disaster number
    row_data = df[df['DisNo.'] == selected_row_str]

    if not row_data.empty:
        #print(f"Row data: {row_data}")
        row_data["geometry"] =  row_data.apply(get_geometries, axis=1)
        
        row_data = row_data.squeeze()

        # Combine the relevant columns into a single storyline with labels
        storyline_parts = [
            f"Key Information: {row_data.get('key information', '')}",
            f"Severity: {row_data.get('severity', '')}",
            f"Key Drivers: {row_data.get('key drivers', '')}",
            f"Main Impacts, Exposure, and Vulnerability: {row_data.get('main impacts, exposure, and vulnerability', '')}",
            f"Likelihood of Multi-Hazard Risks: {row_data.get('likelihood of multi-hazard risks', '')}",
            f"Best Practices for Managing This Risk: {row_data.get('best practices for managing this risk', '')}",
            f"Recommendations and Supportive Measures for Recovery: {row_data.get('recommendations and supportive measures for recovery', '')}"
        ]
        storyline = "\n\n".join(part for part in storyline_parts if part.split(': ')[1])  # Include only non-empty parts
        cleaned_storyline = gpt_story(storyline)
        causal_graph_caption = row_data.get('llama graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_html = plot_cgraph_pyvis(grp)

        # Create the Folium map
        geometry = row_data.get('geometry', None)
        folium_map_html = plot_geometry_folium(geometry, location_name=country, country_name=country)

        # Parse and format the start date
        start_date_str = f"{row_data['Start Year']}-{row_data['Start Month']}-{row_data['Start Day']}"

        # Parse and format the end date
        end_date_str = f"{row_data['End Year']}-{row_data['End Month']}-{row_data['End Day']}"

        return (
            cleaned_storyline,
            causal_graph_html,
            folium_map_html,
            start_date_str,
            end_date_str
        ) 
    else:
        print("No valid data found for the selection.")
        return ('No valid data found.', '<div>No graph available.</div>', '', '', '')

def process_new_node(selected_row_str, new_node):
    if not selected_row_str or selected_row_str == 'Select a Disaster Event':
        print("No valid disaster event selected.")
        return '<div>No graph available.</div>'

    if not new_node:
        print("No new node provided.")
        return '<div>No graph available.</div>'

    print(f"Selected Row: {selected_row_str}, New Node: {new_node}")

    # Filter the dataframe for the selected disaster number
    row_data = df[df['DisNo.'] == selected_row_str]

    if not row_data.empty:
        row_data = row_data.squeeze()
        causal_graph_caption = row_data.get('llama graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        source, relations, target = list(zip(*grp))
        kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

        # Call the generate_new_relations function
        result_df = generate_new_relations(
            graph_df=kg_df,
            new_node=new_node,
            max_combinations_fraction=0.1,
            temperature=0.8,  # Adjust temperature for diversity
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Top-p (nucleus) sampling
            seed=42,  # Optional for reproducibility
            verbose=False  # Optional for debugging
        )

        # Plot the updated graph with the new relations
        source = result_df['source'].astype(str)
        relations = result_df['edge'].astype(str)
        target = result_df['target'].astype(str)
        grp = zip(source, relations, target)
        causal_graph_html = plot_cgraph_pyvis(grp)
        return causal_graph_html
    else:
        print("No valid data found for the selection.")
        return '<div>No graph available.</div>'

def build_interface():
    with gr.Blocks() as interface:
        gr.Markdown(
        """
        # From Complexity to Clarity: Leveraging AI to Decode Interconnected Risks

        Welcome to our Gradio application, developed and maintained by [JRC](https://joint-research-centre.ec.europa.eu/) Units: **E1**, **F7**, and **T5**. This is part of the **EMBRACE Portfolio on Risks**. <br><br>

        **Overview**:  
        This application employs advanced AI techniques like Retrieval-Augmented Generation (RAG) on [EMM](https://emm.newsbrief.eu/) news. It extracts relevant media content on disaster events recorded in [EM-DAT](https://www.emdat.be/), including floods, wildfires, droughts, epidemics, and disease outbreaks. <br><br>

        **How It Works**:  
        For each selected event (filterable by Disaster Type, Country, and Disaster Number), the app:
        - Retrieves pertinent news chunks via the EMM RAG service.
        - Uses multiple LLMs from the [GPT@JRC](https://gpt.jrc.ec.europa.eu/) portfolio to:
            - Extract critical impact data (e.g., fatalities, affected populations).
            - Transform unstructured news into coherent, structured storylines.
            - Build causal knowledge graphs — *impact chains* — highlighting drivers, impacts, and interactions. <br><br>

        **Explore Events**:  
        Use the selectors below to explore events by **Disaster Type**, **Country**, and **Disaster Number (DisNo)**. <br>
        Once an event is selected, the app will display the **causal impact-chain graph**, illustrating key factors and their interrelationships. <br>
        Below the graph, you'll find the **AI-generated narrative**, presenting a structured storyline of the event based on relevant news coverage. <br><br>

        **Outcome**:  
        These outputs offer a deeper understanding of disaster dynamics, supporting practitioners, disaster managers, and policy-makers in identifying patterns, assessing risks, and enhancing preparedness and response strategies.
        """
    )

        # Create dropdowns for Disaster Type, Country, and Disaster Event #
        disaster_type_dropdown = gr.Dropdown(
            choices=[''] + df['Disaster Type'].unique().tolist(),
            label="Select Disaster Type"
        )
        country_dropdown = gr.Dropdown(
            choices=[''],  # Initially empty; will be populated based on disaster type
            label="Select Country"
        )
        row_dropdown = gr.Dropdown(
            choices=[],
            label="Select Disaster Event #",
            interactive=True
        )

        with gr.Column():
            disaster_type_dropdown
            country_dropdown
            row_dropdown

            gr.Markdown("### AI-Generated Storyline:")  # Title
            outputs = [
                gr.Textbox(label="Storyline", interactive=False, lines=10),
                gr.HTML(label="Original Causal Graph"),  # Change from gr.Plot to gr.HTML
                gr.HTML(label="Location Map"),   # Add HTML output for Folium map
                gr.HTML(label="Updated Causal Graph")  # New HTML component for the updated graph
            ]

            # New Radio button for user feedback
            feedback_radio = gr.Radio(
                choices=["Yes", "No"],
                label="According to your expert knowledge, does the graph capture the main relations?",
                interactive=True
            )

            # New section for generating new scenarios
            gr.Markdown("### Generate New Scenarios")  # Subtitle for the new section
            new_node_input = gr.Textbox(
                label="Enter a new variable or factor that might interact with the current graph to generate a plausible scenario:",
                placeholder="e.g., tornado",
                interactive=True
            )

            # Button to save the data
            save_button = gr.Button("Save Data")

            # Button to process new node
            process_button = gr.Button("Add New Node")

        # Update country choices based on selected disaster type
        disaster_type_dropdown.change(
            fn=lambda disaster_type: gr.update(
                choices=[''] + sorted(df[df['Disaster Type'] == disaster_type]['Country'].unique().tolist()),
                value=''
            ),
            inputs=disaster_type_dropdown,
            outputs=country_dropdown
        )

        # Update DisNo. choices based on selected disaster type and country
        country_dropdown.change(
            fn=update_row_dropdown,
            inputs=[disaster_type_dropdown, country_dropdown],
            outputs=row_dropdown
        )

        # Display information based on selected DisNo.
        row_dropdown.change(
            fn=display_info,
            inputs=[row_dropdown, country_dropdown],
            outputs=outputs[:3]  # Do not overwrite the updated graph slot
        )

        # Handle saving data on button click
        save_button.click(
            fn=save_data,
            inputs=[row_dropdown, feedback_radio],
            outputs=[]
        )

        # Handle processing of the new node
        process_button.click(
            fn=process_new_node,
            inputs=[row_dropdown, new_node_input],
            outputs=[outputs[3]]  # Update only the updated graph output
        )

    return interface

app = build_interface()
app.launch()

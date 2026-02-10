import os
import pandas as pd
from datetime import date
import gradio as gr
from pyvis.network import Network
import ast
from openai import OpenAI
import string
from datetime import datetime
import random
import geopandas as gpd
import folium
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.ops import unary_union
import dropbox
from dropbox.exceptions import ApiError
import io 
from itertools import combinations
from typing import Optional, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import inflect
from itertools import combinations
from typing import Optional, Union
import torch
import osmnx as osm


# Replace these with your actual app key and secret
APP_KEY = os.environ['APP_KEY']
APP_SECRET = os.environ['APP_SECRET']
REFRESH_TOKEN = os.environ['REFRESH_TOKEN']


EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
EMM_RETRIEVERS_OPENAI_API_KEY = os.environ['EMM_RETRIEVERS_OPENAI_API_KEY']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#usr_tkn_consose_read = os.environ['usr_tkn_consose_read']

inflect_engine = inflect.engine()

model_id = os.environ['ie_model_id']
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

    
client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)


def geocode_emdat(location):
    def process_geocoding(location_to_geocode):
        try:
            return osm.geocode_to_gdf(location_to_geocode)["geometry"].iloc[0]
        except Exception:
            return None

    geocoded_location = process_geocoding(location)

    if geocoded_location is None:
        print(f"Error geocoding location '{location}'. Trying to correct with GPT-4.")
        response = client1.chat.completions.create(
            model="gpt-4o",
            stream=False,
            messages=[{"role": "user", "content": f"Correct spelling or grammar or substitute with most commonly used location name by Google Maps, give me only the answer in the form 'Country, Location' filled with the corrected Country and Location: '{location}'"}]
        )
        corrected_location = response.choices[0].message.content.strip()
        geocoded_location = process_geocoding(corrected_location)

    return geocoded_location


def get_country_boundary(country_name):
    # Filter the world GeoDataFrame for the country
    country = world[world['NAME'] == country_name]
    if not country.empty:
        # Return the country's geometry
        return country.geometry.iloc[0]
    else:
        # Return None if country not found
        return None

def get_geometries(row):
    country = row['Country']
    locations = row['Locations']
    
    # Return NaN if locations is NaN
    if pd.isna(locations):
        return None
    
    # Get the country's boundary
    country_boundary = get_country_boundary(country)
    
    # If no country boundary is found, return None
    if country_boundary is None:
        return None
    
    locations_list = locations.split(', ')
    
    # Get polygons for each location, ignoring None results
    polygons = [geocode_emdat(f"{country}, {location}") for location in locations_list]
    polygons = [polygon for polygon in polygons if polygon is not None]
    
    # Filter polygons to remove those outside the country boundary
    valid_polygons = [polygon for polygon in polygons if polygon.within(country_boundary)]
    
    # If there are no valid polygons, return None
    if not valid_polygons:
        return None
    
    # Combine them into a single geometry using unary_union
    combined_geometry = unary_union(valid_polygons)
    print(combined_geometry)
    
    return combined_geometry


def singularize(text):
    """Convert a word to its singular form."""
    if inflect_engine.singular_noun(text):
        return inflect_engine.singular_noun(text)
    return text

def is_singular_plural_pair(word1, word2):
    """Check if two words are singular/plural forms of each other."""
    return singularize(word1) == singularize(word2)


def extract_edge_and_clean(row, relations):
    for relation in relations:
        if relation in row['source']:
            row['source'] = row['source'].replace(relation, '').strip()
            row['edge'] = relation
        elif relation in row['target']:
            row['target'] = row['target'].replace(relation, '').strip()
            row['edge'] = relation
    return row

def generate_with_temperature(prompt, temperature=1.0, top_k=50, top_p=0.95, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_texts

def generate_new_relations(
    graph_df: pd.DataFrame,
    new_node: str,
    max_combinations_fraction: float = 0.3,
    num_beams: int = 6,  # Note: Beams are ignored in sampling
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)

    records = graph_df.to_dict('records')
    all_combos = list(combinations(records, 3))
    max_iters = max(1, int(max_combinations_fraction * len(all_combos)))
    num_iters = random.randint(1, max_iters)

    if verbose:
        print(f"Total possible combinations: {len(all_combos)}")
        print(f"Sampling {num_iters} combos")

    all_predictions = []

    for _ in range(num_iters):
        combo = random.choice(all_combos)
        for choice in ('source', 'target'):
            last = combo[-1].copy()
            if choice == 'target':
                prompt_template = f"<extra_id_0> {new_node}."
            else:
                prompt_template = f"{new_node} <extra_id_0>."

            for _ in range(2):
                perm = list(combo[:-1])
                random.shuffle(perm)

                parts = ["If"]
                for row in perm:
                    parts.append(f"{row['source']} {row['edge']} {row['target']},")
                    parts.append("and")
                parts.append(f"{last['source']} {last['edge']} {last['target']},")
                parts.append("then")
                parts.append(prompt_template)
                prompt = " ".join(parts)

                if verbose:
                    print("Generated Prompt:", prompt)

                preds = generate_with_temperature(
                    prompt, temperature, top_k, top_p, max_length
                )

                if verbose:
                    print("Predictions:", preds)

                for text in preds:
                    #print("Generated Prompt:", prompt)
                    #print("text = ", text)
                    #print("last = ", last)
                    all_predictions.append((choice, text, last))
                    
    

    if not all_predictions:
        return graph_df.copy()

    grouped = {}
    for choice, text, last in all_predictions:
        key = (choice, last['source'], last['edge'], last['target'])
        grouped.setdefault(key, []).append(text)

    new_edges = []
    for (choice, src, edge, tgt), texts in grouped.items():
        common = set(texts)
        if not common:
            continue
        for pred in common:
            if choice == 'target':
                new_edges.append({'source': pred, 'edge': None, 'target': new_node})
            else:
                new_edges.append({'source': new_node, 'edge': None, 'target': pred})

    new_df = pd.DataFrame(new_edges).drop_duplicates()

    relations = ['causes', 'prevents']
    new_df = new_df.apply(lambda row: extract_edge_and_clean(row, relations), axis=1)

    result = pd.concat([graph_df, new_df], ignore_index=True)

    result['pair'] = result.apply(lambda x: tuple(sorted([x['source'], x['target']])), axis=1)
    result = result.drop_duplicates(subset=['pair'])
    result = result[result['source'] != result['target']]
    result = result.drop(columns=['pair'])
    # Remove duplicates based on plural/singular forms
    result['source_singular'] = result['source'].apply(singularize)
    result['target_singular'] = result['target'].apply(singularize)
    result = result.drop_duplicates(subset=['source_singular', 'edge', 'target_singular'])
    result = result[result['source'] != result['target']]
    result = result.drop(columns=['source_singular', 'target_singular'])

    return result

# Function to get a Dropbox client, refreshing the token if needed
def get_dropbox_client():
    try:
        # Create a Dropbox client using the refresh token
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=REFRESH_TOKEN,
            app_key=APP_KEY,
            app_secret=APP_SECRET
        )
        return dbx
    except Exception as e:
        print(f"Error creating Dropbox client: {e}")
        return None

    
client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat_adinet_idmc.csv", sep=',', header=0, dtype=str, encoding='utf-8')

#world = gpd.read_file('https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/ne_110m_admin_0_countries.shp')


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
        model='llama-3.3-70b-instruct',
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
    dbx = get_dropbox_client()
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
    dbx = get_dropbox_client()
    if not dbx:
        print("Failed to create Dropbox client.")
        return

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


#def append_to_csv_on_dropbox(new_content, dropbox_path):
#    dbx = dropbox.Dropbox(ACCESS_TOKEN)

#    try:
        # Try to download existing file content
#        metadata, res = dbx.files_download(dropbox_path)
#        existing_content = res.content.decode('utf-8')
#    except ApiError as e:
        # If file not found, start with empty content
#        if e.error.is_path() and e.error.get_path().is_not_found():
#            existing_content = ''
#        else:
#            print(f"Error downloading file: {e}")
#            return

    # Append new content without header if file already exists
#    if existing_content:
#        new_content_lines = new_content.splitlines()
#        new_content_without_header = '\n'.join(new_content_lines[1:])
#        combined_content = existing_content.rstrip('\n') + '\n' + new_content_without_header
#    else:
#        combined_content = new_content

#    try:
        # Upload combined content back to Dropbox (overwrite)
#        dbx.files_upload(combined_content.encode('utf-8'), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
#        print(f"Appended and uploaded to {dropbox_path} successfully!")
#    except Exception as e:
#        print(f"Error uploading file: {e}")

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
    
    return "Thank you, your answer has been recorded! This will help make AI more reliable."


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
        #row_data["geometry"] =  row_data.apply(get_geometries, axis=1)
        
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
        #cleaned_storyline = gpt_story(storyline)
        cleaned_storyline = storyline
        causal_graph_caption = row_data.get('llama graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_html = plot_cgraph_pyvis(grp)

        # Create the Folium map
        #geometry = row_data.get('geometry', None)
        #folium_map_html = plot_geometry_folium(geometry, location_name=country, country_name=country)

        # Parse and format the start date
        #start_date_str = f"{row_data['Start Year']}-{row_data['Start Month']}-{row_data['Start Day']}"

        # Parse and format the end date
        #end_date_str = f"{row_data['End Year']}-{row_data['End Month']}-{row_data['End Day']}"

        return (
            cleaned_storyline,
            causal_graph_html,
        #    folium_map_html,
        #    start_date_str,
        #    end_date_str
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
        This application employs advanced AI techniques like Retrieval-Augmented Generation (RAG) on [EMM](https://emm.newsbrief.eu/) news. It extracts relevant media content on disaster events recorded in [EM-DAT](https://www.emdat.be/), [IDMC](https://www.internal-displacement.org/), and [ADINet](https://adinet.ahacentre.org/), including floods, wildfires, droughts, epidemics, and disease outbreaks. <br><br>

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
                #gr.HTML(label="Location Map"),   # Add HTML output for Folium map
                gr.HTML(label="Updated Causal Graph")  # New HTML component for the updated graph
            ]

            # New Radio button for user feedback
            feedback_radio = gr.Radio(
                choices=[
                    "Fully Correct",
                    "Mostly Correct",
                    "Partially Incorrect",
                    "Incorrect",
                    "Difficult to Judge"
                ],
                label="According to your expert knowledge, evaluate the graph based on the following descriptions:",
                interactive=True
            )

            # Add descriptions for each option
            gr.Markdown("""
            - **Fully Correct**: All relevant nodes are accurately identified. The relationships among these nodes, including their directions (causes or prevents), are correct.
            - **Mostly Correct**: The majority of relevant nodes are identified. However, some relationships may be missing or incorrect, such as a missing link or an inaccurate parent-child relationship. Despite these minor issues, the overall structure is largely accurate.
            - **Partially Incorrect**: Some important nodes are missing and there are errors in the directions of relationships. However, the graph still captures some of the main characteristics of the intended relationships.
            - **Incorrect**: The majority of links are incorrect, and many relevant nodes are either missing or inaccurately represented.
            - **Difficult to Judge**: The graph is ambiguous or lacks sufficient context for accurate assessment.
            """)

            # New section for generating new scenarios
            gr.Markdown("### Generate New Scenarios")  # Subtitle for the new section
            new_node_input = gr.Textbox(
                label="Enter a new variable or factor that might interact with the current graph to generate a plausible scenario:",
                placeholder="e.g., tornado",
                interactive=True
            )

            # Button to save the data
            save_button = gr.Button("Save Your Answer")

            # Feedback message
            feedback_message = gr.Markdown("", visible=True)

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
            outputs=outputs[:2]  # Do not overwrite the updated graph slot, change to 3
        )

        # Handle saving data on button click
        save_button.click(
            fn=save_data,
            inputs=[row_dropdown, feedback_radio],
            outputs=[feedback_message],
            #api_name="Save Answer"
        )

        # Handle processing of the new node
        process_button.click(
            fn=process_new_node,
            inputs=[row_dropdown, new_node_input],
            outputs=[outputs[2]]  # Update only the updated graph output, change to 3
        )

    return interface

app = build_interface()
app.launch()

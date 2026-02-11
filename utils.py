import pandas as pd
import numpy as np
import time
import pycountry
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
import re
import json
from openai import OpenAI
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import osmnx as osm
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import textwrap


world = gpd.read_file('./data/ne_110m_admin_0_countries.shp')


with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    TOKEN = config['EMM_RETRIEVERS_OPENAI_API_KEY']


client1 = OpenAI(
    api_key=TOKEN,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

#model = "nous-hermes-2-mixtral-8x7b-dpo"
model = "llama-3.3-70b-instruct"

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
    
    return combined_geometry


def fact_check(disaster, month, year, location, page_content):
    """
    Checks if the document content is relevant to the specified disaster event using a language model.

    :param disaster: Name of the disaster event
    :param month: Month of the disaster event.
    :param year: Year of the disaster event.
    :param location: Specific location affected by the disaster.
    :param page_content: The content of the document.
    :return: "Yes" if relevant, otherwise "No".
    """
    # Construct the prompt for the LLM
    prompt = (
        f"Is the following document referring to the {disaster} disaster "
        f"that occurred in {location} during {month} {year}? "
        f"Please answer only with 'Yes' or 'No' without adding anything else.\n\n"
        f"Document Content: {page_content}"
    )

    # Call the language model with the prompt
    completion = client1.chat.completions.create(
        model=model,  # Replace with the appropriate model for your use case
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    #print(completion.choices[0].message.content.strip())
    return completion.choices[0].message.content.strip()


def extract_triplets(nested_list):
    def traverse_structure(structure):
        triplets = []
        for item in structure:
            if isinstance(item, list):
                if len(item) == 3 and (item[1] == 'causes' or item[1] == 'prevents'):
                    # It's a valid triplet
                    triplets.append(item)
                else:
                    # Recursively traverse deeper if it’s a nested list
                    triplets.extend(traverse_structure(item))
        return triplets

    # Start the recursive extraction
    return traverse_structure(nested_list)

def needs_correction(relationships):
    # Check for excessive nesting or malformed entries
    return any(isinstance(item, list) and len(item) != 3 for item in relationships)

def extract_unique_nodes(relationships):
    # Extract triplets and initialize a set for unique nodes
    triplets = extract_triplets(relationships)
    unique_nodes = set()

    for triplet in triplets:
        try:
            # Ensure elements are hashable types like strings
            node1, node2 = triplet[0], triplet[2]
            unique_nodes.add(node1)
            unique_nodes.add(node2)
        except TypeError as e:
            print(f"Skipping malformed nodes in row {index}: {triplet}. Error: {e}")

    return list(unique_nodes)

def balance_brackets(s):
    s = s.replace('}', ']')
    s = s.rstrip(",]")
    s = re.sub(r'([a-zA-Z]), \[', r'\1"], [', s)
    s = s.split("using shorter")[0].strip().rstrip('.')  # Use lowercase "using shorter"
    open_count = s.count('[')
    close_count = s.count(']')

    if open_count > close_count:
        s += ']' * (open_count - close_count)
    elif close_count > open_count:
        s = '[' * (close_count - open_count) + s

    return s

def clean_structure(s):
    s = s.split("however")[0]  # Use lowercase "however"
    s = re.sub(r'\]\s+and\s+\[', '], [', s)
    s = re.sub(r'\]\n\n\[', '], [', s)
    s = s.rsplit(']', 1)[0] + ']'
    return s

def remove_duplicate_keywords(relationships):
    cleaned_relationships = []
    for relation in relationships:
        cleaned_relation = []
        previous_word = None
        for word in relation:
            if word != previous_word:
                cleaned_relation.append(word)
            previous_word = word
        cleaned_relationships.append(cleaned_relation)
    return cleaned_relationships

def extract_relationships_from_string(s):
    lines = s.strip().split('\n')
    relationships = []

    for line in lines:
        line = re.sub(r'^[-\d.]+\s*', '', line.strip())
        if not line:
            continue

        if 'causes' in line:
            parts = line.split('causes')
            if len(parts) == 2:
                cause, effect = parts
                relationships.append([cause.strip(), 'causes', effect.strip()])
        elif 'prevents' in line:
            parts = line.split('prevents')
            if len(parts) == 2:
                prevention, effect = parts
                relationships.append([prevention.strip(), 'prevents', effect.strip()])

    relationships = [[elem.strip('", ') if isinstance(elem, str) else elem for elem in relation] for relation in relationships]
    return relationships

def extract_list_from_string(s):
    s = s.lower()  # Convert to lowercase
    first_bracket_index = s.find('[')
    if first_bracket_index != -1:
        s = s[first_bracket_index:]

    s = balance_brackets(s)
    s = clean_structure(s)

    try:
        relationships = json.loads(s)
        relationships = [[elem.strip('", ') if isinstance(elem, str) else elem for elem in relation] for relation in relationships]
    except json.JSONDecodeError:
        relationships = extract_relationships_from_string(s)

    return remove_duplicate_keywords(relationships)

def transform_triplets(relationships):
    def clean_element(element):
        if isinstance(element, list):
            element = ' '.join(element)
        return re.sub(r'[^a-zA-Z\s-]', '', element).strip()

    transformed_list = [
        (clean_element(source), clean_element(relation), clean_element(target))
        for triplet in relationships if len(triplet) == 3
        for source, relation, target in [triplet]
    ]

    # Filter to only include triplets with 'causes' or 'prevents' as the relation
    filtered_list = [
        triplet for triplet in transformed_list
        if triplet[1] in {"causes", "prevents"}
    ]

    # Remove triplets with any element longer than 50 characters
    filtered_list = [
        triplet for triplet in filtered_list
        if all(len(element) <= 50 for element in triplet)
    ]
    
    return filtered_list

def process_graph(s):
    try:
        if isinstance(s, str):
            relationships = extract_list_from_string(s)
            return transform_triplets(relationships)
        else:
            return None
    except Exception as e:
        return None 


def iso3_to_iso2(iso3_code):
    # Iterate through countries in pycountry and find a match for the ISO3 code
    country = pycountry.countries.get(alpha_3=iso3_code)
    if country:
        return country.alpha_2
    else:
        return None  # Return None if no match is found

def generate_date_ranges(start_dt, num_weeks=4):
    start_dt = pd.to_datetime(start_dt)
    date_ranges = []
    for i in range(num_weeks):
        start = start_dt + pd.Timedelta(weeks=i)
        end = start + pd.Timedelta(weeks=1)
        date_ranges.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    return date_ranges



def process_documents(docs, iso2, country, disaster, month, year, location, format_fn=format_doc_minimal, sleep_interval=1, **kwargs):
    """
    Filters and formats documents using specified criteria and formatting functions.

    :param docs: List of documents to process.
    :param iso2: ISO 2-letter country code for filtering.
    :param country: Country name to check in the title.
    :param disaster: Name of the disaster event.
    :param location: Specific location affected by the disaster.
    :param format_fn: Function to format documents.
    :param sleep_interval: Time to sleep between fact_check calls to manage rate limits.
    :param kwargs: Additional arguments for the formatting function.
    :return: Tuple of formatted string of filtered documents and count of relevant documents.
    """
    
    # Initial filtering based on country code and title
    filtered_docs = [
        entry for entry in docs
        if entry['metadata']['source']['country'] == iso2 or
           country in entry['metadata']['title']
    ]

    relevant_docs = []
    
    # Further filter using the fact_check function
    for entry in filtered_docs:
        if fact_check(disaster, month, year, location, entry['page_content']) == "Yes":
            #print(disaster, month, year, location)
            #print(entry['page_content'])
            relevant_docs.append(entry)
        time.sleep(sleep_interval)  # Add a sleep interval to avoid hitting rate limits
    
    num_relevant_docs = len(relevant_docs)
    print("Num filtered docs = ", num_relevant_docs)

    # Format the relevant documents
    formatted_docs = format_docs(relevant_docs, doc_fn=format_fn, **kwargs)
    
    return formatted_docs, num_relevant_docs



def add_sections_as_columns(row, txt, graph):
    # Define the column names
    column_names = [
        "Key information",
        "Severity",
        "Key drivers",
        "Main impacts, exposure, and vulnerability",
        "Likelihood of multi-hazard risks",
        "Best practices for managing this risk",
        "Recommendations and supportive measures for recovery"
    ]

    # Create a dictionary to hold the extracted content
    content_dict = {col.lower(): "" for col in column_names}

    # Create a regex pattern to match column names directly
    pattern = '|'.join([re.escape(col.lower()) for col in column_names])

    # Use regex to find all matches and their positions in the text
    matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, txt.lower())]

    # Iterate over the matches and extract the text for each column
    for i, (start, end, col_name) in enumerate(matches):
        # Determine the start of the content
        content_start = end
        # Determine the end of the content
        content_end = matches[i + 1][0] if i + 1 < len(matches) else len(txt)
        # Extract the content and strip excess whitespace
        content = txt[content_start:content_end].strip()
        # Replace newlines and nested bullet points with spaces
        content = re.sub(r'\n\s*-\s*', '; ', content).replace('\n', ' ')
        # Store the content in the dictionary
        content_dict[col_name] = content

    # Convert the row to a DataFrame
    row_df = pd.DataFrame([row])

    # Add the extracted content to the row as new columns
    for title, content in content_dict.items():
        row_df[title] = content
    
    # Add the graph
    row_df["causal graph"] = graph

    return row_df


def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    cleaned_text = re.sub(r'^[^\w]+|[^\w]+$', '', text)

    return cleaned_text.strip()


def process_storyline(row):
    # Clean each relevant column in the row
    row['key information'] = clean_text(row['key information'])
    row['severity'] = clean_text(row['severity'])
    row['key drivers'] = clean_text(row['key drivers'])
    row['main impacts, exposure, and vulnerability'] = clean_text(row['main impacts, exposure, and vulnerability'])
    row['likelihood of multi-hazard risks'] = clean_text(row['likelihood of multi-hazard risks'])
    row['best practices for managing this risk'] = clean_text(row['best practices for managing this risk'])
    row['recommendations and supportive measures for recovery'] = clean_text(row['recommendations and supportive measures for recovery'])
    
    # Combine cleaned text for checking purposes
    combined_text = (
        f"key information: {row['key information']}\n"
        f"severity: {row['severity']}\n"
        f"key drivers: {row['key drivers']}\n"
        f"main impacts, exposure, and vulnerability: {row['main impacts, exposure, and vulnerability']}\n"
        f"likelihood of multi-hazard risks: {row['likelihood of multi-hazard risks']}\n"
        f"best practices for managing this risk: {row['best practices for managing this risk']}\n"
        f"recommendations and supportive measures for recovery: {row['recommendations and supportive measures for recovery']}"
    )
    
    # Count occurrences of 'unknown' in any case
    unknown_count = combined_text.lower().count('unknown')
    
    # Return the row if 'unknown' appears less than 5 times
    if unknown_count < 5:
        return row
    else:
        return None
    
def custom_sum(x, y):
    if x is None and y is None:
        return np.nan
    elif x is None:
        return y
    elif y is None:
        return x
    else:
        return x + y


def extract_disaster_info(disaster, month, year, country, formatted_docs):
    """
    Extracts specific disaster information from formatted documents using a language model
    and ensures the JSON format is correct.

    :param disaster: Name of the disaster event.
    :param month: Month of the disaster event.
    :param year: Year of the disaster event.
    :param country: The country where the disaster occurred.
    :param formatted_docs: The formatted content of the documents.
    :return: A dictionary with extracted information or None if not available.
    """
    json_template = {
        "People affected": None,
        "Fatalities": None,
        "Economic losses": None,
        "Locations": None
    }

    # Construct the prompt for the LLM
    prompt = (
        f"You are an expert in disaster event analysis. Based on the content related to the {disaster} disaster "
        f"that occurred in {country} during {month} {year}, please fill in the following JSON template. "
        f"For 'Locations', list all mentioned cities or provinces only within {country}, ignoring any outside of {country}, and separate them by commas. "
        f"For 'People affected', 'Fatalities', and 'Economic losses', return only the total amount according to the Document Content; do not include any additional words or text. "
        f"Use 'None' for any field where the information is not available.\n\n"
        f"Document Content: {formatted_docs}\n\n"
        f"JSON Template:\n{json_template}"
    )

    # Call the language model with the prompt
    completion = client1.chat.completions.create(
        model=model,  # Replace with your model
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Extract the response content
    response_content = completion.choices[0].message.content.strip()

    # Ensure JSON format by adding missing quotes to keys and string values
    def ensure_json_format(content):
        # Define expected keys
        expected_keys = ["People affected", "Fatalities", "Economic losses", "Locations"]

        # Regular expression pattern to find possible key-value pairs
        key_value_pattern = re.compile(r'(\b(?:' + '|'.join(map(re.escape, expected_keys)) + r')\b)\s*:\s*([^,\}\n]+)', re.DOTALL)

        def add_quotes_to_value(match):
            key, value = match.groups()
            # Check if value is not already quoted
            if not (value.startswith('"') and value.endswith('"')):
                # Add quotes around the value if it's not a number or None
                if value.strip().lower() != 'none' and not re.match(r'^\d+(\.\d+)?$', value.strip()):
                    value = f'"{value.strip()}"'
            return f'"{key}": {value}'

        # Add quotes to keys and string values
        json_like_string = key_value_pattern.sub(add_quotes_to_value, content)
        return json_like_string

    # Process the response content to ensure JSON format
    json_like_string = ensure_json_format(response_content)

    # Use a regular expression to extract JSON from the processed content
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    match = json_pattern.search(json_like_string)
    
    if match:
        try:
            # Parse the extracted JSON
            extracted_data = json.loads(match.group())
            return extracted_data
        except json.JSONDecodeError as e:
            # Print the response for debugging purposes
            print("JSON decoding failed. Error:", e)
            print("Processed content:", match.group())
            return None
    else:
        # Print the response for debugging purposes
        print("No JSON found in the response. Processed content:", json_like_string)
        return None


def plot_cgraph(kg_df):
    # Create a directed graph from a dataframe
    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

    # Define a color mapping for edge types
    edge_colors_dict = {
        "causes": "red",
        "prevents": "green",
    }

    # Extract the colors for each edge in the graph based on the 'edge' attribute
    edge_color_list = [edge_colors_dict[G[u][v][key]['edge']] for u, v, key in G.edges(keys=True)]

    # Draw the graph
    plt.figure(figsize=(16, 16))  # Increase the figure size

    # Compute the spring layout
    pos = nx.spring_layout(G, k=2, iterations=100)  # Adjust k value for more spacing

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=8500, alpha=0.7)  # Increased size and adjusted transparency

    # Draw edges with arrows and custom styling
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_color_list,
        arrows=True,
        arrowsize=20,
        width=2,
        connectionstyle='arc3,rad=0.1',
        arrowstyle='-|>',  # Use a different arrow style
        min_target_margin=46  # Add margin to ensure arrows stop at node borders
    )

    # Preprocess node labels to insert line breaks for long labels
    labels_with_linebreaks = {node: '\n'.join(node.split(' ')) for node in G.nodes()}

    # Define node label options to prevent overlap
    node_label_options = {
        'font_size': 16,  # Increased font size
        'font_weight': 'bold',
        'verticalalignment': 'center',
        'horizontalalignment': 'center'
    }

    # Draw node labels with the modified labels
    nx.draw_networkx_labels(G, pos, labels=labels_with_linebreaks, **node_label_options)

    # Create a legend for the edge colors
    legend_elements = [Line2D([0], [0], color=color, label=edge_type, lw=2) for edge_type, color in edge_colors_dict.items()]
    plt.legend(handles=legend_elements, loc='best', fontsize=22)  # Increased legend font size

    # Set the aspect ratio of the plot to equal and adjust margins
    plt.gca().set_aspect('equal', adjustable='box')
    plt.margins(x=0.1, y=0.1)

    # Turn off the axis lines and labels
    plt.axis('off')

    # Show the plot with a tight layout
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig("knowledge_graph.png", format="PNG", bbox_inches='tight', dpi=500)

    # Display the plot
    plt.show()



# --------------------------------------------------------------
# Helper – strip asterisks from a string
# --------------------------------------------------------------
def _strip_asterisks(txt) -> str:
    """Remove every ``*`` from *txt* (handles NaN safely)."""
    if pd.isna(txt):
        return "‑"
    return re.sub(r"\*", "", str(txt))


# --------------------------------------------------------------
# Factsheet – cleaned text, bold titles, no overlap
# --------------------------------------------------------------
def plot_factsheet_clean(
    em: pd.DataFrame,
    disno: str,
    fact_columns: list | None = None,
    wrap_width: int = 45,
    figsize: tuple = (8, 11),                 # portrait rectangle
    panel_bg: str = "#f0f0f0",                # light‑grey background
    panel_edge: str = "#cccccc",
    title_fontsize: int = 14,
    value_fontsize: int = 12,
    extra_gap: float = 0.09,                 # extra space between sections (axes fraction)
    save_path: str | None = None,
):
    """
    Create a vertical factsheet for a given disaster (DisNo.).
    * Column names are rendered **bold**.
    * All asterisk characters are removed from the values.
    * The whole box has a light‑grey background.
    * Vertical spacing is computed dynamically so that long paragraphs
      never overwrite the next section.
    Returns the Matplotlib ``Figure`` object.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Which columns to show?
    # ------------------------------------------------------------------
    default_cols = [
        "key information",
        "severity",
        "key drivers",
        "main impacts, exposure, and vulnerability",
        "likelihood of multi-hazard risks",
        "best practices for managing this risk",
        "recommendations and supportive measures for recovery",
    ]
    if fact_columns is None:
        fact_columns = default_cols

    # keep only those that exist in the DataFrame
    fact_columns = [c for c in fact_columns if c in em.columns]
    if not fact_columns:
        raise ValueError("None of the requested fact‑sheet columns exist in `em`.")

    # ------------------------------------------------------------------
    # 2️⃣  Pull the row that matches the disaster id
    # ------------------------------------------------------------------
    try:
        fact_row = em.loc[em["DisNo."] == disno, fact_columns].iloc[0]
    except IndexError:
        raise ValueError(f"DisNo. '{disno}' not found in the supplied `em` DataFrame.")

    # ------------------------------------------------------------------
    # 3️⃣  Prepare the figure / axes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")                     # hide ticks / spines

    # ------------------------------------------------------------------
    # 4️⃣  Some geometry helpers (convert font size → axes‑fraction height)
    # ------------------------------------------------------------------
    # Height of the figure in inches
    fig_h_in = figsize[1]

    # Approximate height of one line of text in axes fraction.
    #   (fontsize points) / (72 points per inch) = inches of line height
    #   divide by figure height (in inches) → fraction of the axes.
    line_height = (value_fontsize / 72) / fig_h_in

    # Title line is a little larger
    title_line_height = (title_fontsize / 72) / fig_h_in

    # Starting vertical position (top of the axes)
    cur_y = 0.98          # a little below the very top (so the box edge is visible)

    # ------------------------------------------------------------------
    # 5️⃣  Draw each section (title in bold, value below)
    # ------------------------------------------------------------------
    left_margin = 0.02    # left‑hand margin inside the axes (fraction)

    for col in fact_columns:
        # ---- title (bold) -------------------------------------------------
        ax.text(
            left_margin,
            cur_y,
            col.title(),
            fontsize=title_fontsize,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
            color="black",
        )
        cur_y -= title_line_height + extra_gap           # move below the title

        # ---- value (cleaned, wrapped) ------------------------------------
        raw_val = fact_row[col]
        cleaned = _strip_asterisks(raw_val)
        wrapped = textwrap.fill(cleaned, width=wrap_width)

        # Number of lines the wrapped text occupies
        n_lines = wrapped.count("\n") + 1

        ax.text(
            left_margin,
            cur_y,
            wrapped,
            fontsize=value_fontsize,
            verticalalignment="top",
            horizontalalignment="left",
            color="black",
        )

        # Move the cursor down for the next *title*.
        #   – value part occupies n_lines * line_height
        #   – add a small gap after the block
        cur_y -= n_lines * line_height + extra_gap

    # ------------------------------------------------------------------
    # 6️⃣  Draw the surrounding rounded box (light‑grey background)
    # ------------------------------------------------------------------
    # The invisible text trick attaches a bbox that covers the whole axes.
    ax.text(
        0,
        0,
        "",
        bbox=dict(
            facecolor=panel_bg,
            edgecolor=panel_edge,
            boxstyle="round,pad=0.8",
            linewidth=1,
        ),
        transform=ax.transAxes,
        zorder=-1,
    )

    # ------------------------------------------------------------------
    # 7️⃣  Save / show
    # ------------------------------------------------------------------
    if save_path is None:
        save_path = f"factsheet_{disno}.png"

    fig.savefig(save_path, format="PNG", bbox_inches="tight", dpi=500)
    plt.show()
    return fig

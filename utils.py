import pandas as pd
import numpy as np
import time
import pycountry
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
import re
import json
from openai import OpenAI

with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    TOKEN = config['EMM_RETRIEVERS_OPENAI_API_KEY']


client1 = OpenAI(
    api_key=TOKEN,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

#model = "nous-hermes-2-mixtral-8x7b-dpo"
model = "llama-3.3-70b-instruct"

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


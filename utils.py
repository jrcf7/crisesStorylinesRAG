import pandas as pd
import pycountry
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
import re


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

def process_documents(docs, iso2, country, format_fn=format_doc_minimal, **kwargs):
    """
    Filters and formats documents using specified criteria and formatting functions.

    :param docs: List of documents to process.
    :param iso2: ISO 2-letter country code for filtering.
    :param country: Country name to check in the title.
    :param format_fn: Function to format documents.
    :param kwargs: Additional arguments for the formatting function.
    :return: Formatted string of filtered documents.
    """
    # Apply the filtering criterion
    filtered_docs = [
        entry for entry in docs
        if entry['metadata']['source']['country'] == iso2 or
           country in entry['metadata']['title']
    ]
    
    # Format the filtered documents
    return format_docs(filtered_docs, doc_fn=format_fn, **kwargs)

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
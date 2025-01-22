import pandas as pd
from client_v1.settings import EmmRetrieversSettings
from utils import iso3_to_iso2

EMM_RETRIEVERS_API_BASE="https://api.emm4u.eu/retrievers/v1"
EMM_RETRIEVERS_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlMS11c2VyIiwiZXhwIjoxNzM3MDMyMzMxLCJyb2xlcyI6bnVsbCwiZW1tX2FkbWluIjpmYWxzZSwiZW1tX2FsbG93X2luZGljZXMiOlsibWluZV9lX2VtYi1yYWdfbGl2ZSIsIm1pbmVfZV9lbWItcmFnX2xpdmVfdGVzdF8wMDEiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8qIiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNCIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTUiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE2IiwibWluZV9lX2VtYjE2LWUxZjdfcHJvZDRfMjAxNyIsIm1pbmVfZV9lbWIxNi1lMWY3X3Byb2Q0XzIwMTgiLCJtaW5lX2VfZW1iMTYtZTFmN19wcm9kNF8yMDE5Il0sImVtbV9hbGxvd19jbHVzdGVycyI6WyJyYWctb3MiXX0.My2po1YcbAWlGMG6fvgqMdj3qvXw-vNETFVVrOxCEBQ"
EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
EMM_RETRIEVERS_OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQyNjA3M2JiLTllYWQtNGRmYy04MmU5LTY3NWYyNGFlZjQzOSIsImlzcyI6ImdwdGpyYyIsImlhdCI6MTczNDUxNDI0NiwiZXhwIjoxNzYxOTU1MjAwLCJpc19yZXZva2VkIjpmYWxzZSwiYWNjb3VudF9pZCI6ImIyNGJiZGEwLWY5YjEtNGFkNS1hNGU2LWYyYjE2MzA5ZGI5ZiIsInVzZXJuYW1lIjoiTWljaGVsZS5ST05DT0BlYy5ldXJvcGEuZXUiLCJwcm9qZWN0X2lkIjoiSU5GT1JNIiwiZGVwYXJ0bWVudCI6IkpSQy5FLjEiLCJxdW90YXMiOlt7Im1vZGVsX25hbWUiOiJncHQtNG8iLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTM1LXR1cmJvLTExMDYiLCJleHBpcmF0aW9uX2ZyZXF1ZW5jeSI6ImRhaWx5IiwidmFsdWUiOjQwMDAwMH0seyJtb2RlbF9uYW1lIjoiZ3B0LTQtMTEwNiIsImV4cGlyYXRpb25fZnJlcXVlbmN5IjoiZGFpbHkiLCJ2YWx1ZSI6NDAwMDAwfV0sImFjY2Vzc19ncm91cHMiOlt7ImFjY2Vzc19ncm91cCI6ImdlbmVyYWwifV19.rMVr1vKb_HUvgowbeH8LhC9g7ZICcvWzDGgaY2yr-8o"


settings = EmmRetrieversSettings(
        API_BASE=EMM_RETRIEVERS_API_BASE,
        API_KEY=EMM_RETRIEVERS_API_KEY,
        OPENAI_API_BASE_URL=EMM_RETRIEVERS_OPENAI_API_BASE_URL,
        OPENAI_API_KEY=EMM_RETRIEVERS_OPENAI_API_KEY,
        LANGCHAIN_API_KEY="your_langchain_api_key"
    )


emdat = pd.read_excel("./data/public_emdat_1419.xlsx")
emdat['Start Month'] = emdat['Start Month'].fillna(1).astype(int)
emdat['Start Day'] = emdat['Start Day'].fillna(1).astype(int)
emdat['start_dt'] = pd.to_datetime(emdat['Start Year'].astype(str) + '-' +
                                   emdat['Start Month'].astype(str).str.zfill(2) + '-' +
                                   emdat['Start Day'].astype(str).str.zfill(2))
emdat['start_dt'] = emdat['start_dt'].dt.strftime('%Y-%m-%d')

mnts = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December"
}


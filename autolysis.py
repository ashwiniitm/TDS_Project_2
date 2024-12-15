# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet>=5.2.0",
#     "matplotlib>=3.10.0",
#     "openai>=1.57.4",
#     "pandas>=2.2.3",
#     "python-dotenv>=1.0.1",
#     "requests>=2.32.3",
#     "scikit-learn>=1.6.0",
#     "seaborn>=0.13.2",
# ]
# ///


import sys
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import requests
import json
import chardet
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import base64

def detect_file_encoding(file_path):
    """Detect the encoding of the input file."""
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
    return result['encoding']

def load_dataset(file_path, encoding):
    """Load the dataset with the given encoding."""
    return pd.read_csv(file_path, encoding=encoding)

def initialize_api_details():
    """Initialize API details from environment variables."""
    load_dotenv()
    return {
        "token": os.getenv("AIPROXY_TOKEN"),
        "base_url": "https://aiproxy.sanand.workers.dev/openai/v1",
    }

def prepare_feature_relevance_prompt(dataset):
    encoder = LabelEncoder()
    encodedDataset = dataset.copy(deep=True)
    for col in encodedDataset.columns:
        if encodedDataset[col].dtype == "object":
            encodedDataset[col] = encoder.fit_transform(encodedDataset[col])
    correlation_matrix = encodedDataset.corr()
    columns = [col for col in dataset.columns]
    datatypes = [dataset[col].dtype for col in dataset.columns]
    summary = dataset.describe()

    prompt = f"""
You are an expert in data analysis.
Assume the dataset is called dataset.

The following information is provided:

Correlation Matrix: {correlation_matrix}
Column Names: {columns}
Original Data Types: {datatypes}
Basic Summary: {summary}

Your task is to:

Make an educated guess about the target variable based on logical reasoning.
Remove irrelevant features only if the number of features is more than 10.
Remove features by analyzing the correlation, data types, and uniqueness of each column.
Use the correlation matrix and summary given to you.
Drop features with high missingness, low correlation, or high uniqueness (like IDs).
The output must be only Python code, with no comments or explanations.
Do not output anything other than the code.
Ensure the code cleanly removes the irrelevant features from the DataFrame named "dataset".
The new dataset is also stored as "dataset".
"""
    return prompt

def encode_image(image_path):
    """ Function to encode images in local environment. """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai_api(api_details, endpoint, prompt, functions=None, use_vision = False):
    """Make a request to the OpenAI API and return the response."""
    url = f"{api_details['base_url']}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_details['token']}"
    }
    if use_vision:
        image_paths = ["1.png", "2.png", "3.png"]
        encoded_images = {image: encode_image(image) for image in image_paths}
        data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze these images and integrate the insights into the README."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['1.png']}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['2.png']}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['3.png']}"
                    }
                }
            ]
        }
    ]
}
    else:
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }

    if functions:
        data["functions"] = functions
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")

def execute_code(code):
    """Execute the Python code returned by the API."""
    try:
        exec(code, globals())
        print("Code Executed Successfully")
    except Exception as e:
        print("Error during code execution:", e)

def write_into_file(content):
    """ Function to write the content generated into the README file. """
    with open("README.md", "w") as file:
        file.write(content)

def handle_null_values(dataset, threshold=0.5, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """ Handles null values in a dataset using a fixed heuristic for numeric and categorical variables. """
    null_ratios = dataset.isnull().mean()

    # Handle columns with a high proportion of missing values
    columns_to_drop = null_ratios[null_ratios > threshold].index
    dataset = dataset.drop(columns=columns_to_drop)

    # Separate numeric and categorical columns
    numeric_cols = dataset.select_dtypes(include=['number']).columns
    categorical_cols = dataset.select_dtypes(exclude=['number']).columns

    # Impute numeric columns
    if numeric_cols.any():
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

    # Impute categorical columns
    if categorical_cols.any():
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

def feature_relevance(dataset, api_details):
    """ Function to dynamically adjust the relevance of the features using statistical measures. """
    prompt = prepare_feature_relevance_prompt(dataset)
    response = call_openai_api(api_details, "chat/completions", prompt)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("python").strip()
    print(clean_code)
    execute_code(clean_code)

def generate_boxplots(columns, dataset):
    """Generate box plots for the specified columns to detect outliers."""
    plt.figure(figsize=(10, 6))
    dataset[columns].boxplot()
    plt.title('Box Plots for Outlier Detection')
    plt.show()

def remove_outliers(dataset, columns):
    """Remove outliers from the dataset based on the IQR method."""
    cleaned_data = dataset.copy()
    for column in columns:
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]
    return cleaned_data

def handle_outliers(dataset, columns, api_details):
    """Generate box plots, remove outliers, and return the cleaned dataset."""
    functions = [
        {
            "name": "generate_boxplots",
            "description": "Generate box plots for outlier detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "dataset": {"type": "object"}
                }
            }
        },
        {
            "name": "remove_outliers",
            "description": "Remove outliers from the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "object"},
                    "columns": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    ]

    prompt = f"""
    Name of the dataset = "dataset"
    I have a dataset with the following numerical columns: {', '.join(columns)}.
    Perform the following tasks:
    1. Generate box plots to visualize potential outliers.
    2. Remove identified outliers based on your analysis.
    """
    response = call_openai_api(api_details, "chat/completions", prompt, functions)
    cleaned_data = remove_outliers(dataset, columns)
    return cleaned_data

def visualize_data(dataset, columns, datatypes, api_details, correlation_matrix):
    """Generate visualizations for the dataset using Seaborn and Matplotlib."""

    prompt = f"""
    You are a Python Data Analyst.
    Given the following column names: {columns}
    And their respective datatypes: {datatypes}
    The Correlation between each Column: {correlation_matrix}
    Assume that the dataset is already imported.
    *IMPORTANT* Name of the dataset: "dataset"

    Generate Python code using Seaborn to:
    1. Convert any date column into datetime using "to_datetime" method in pandas.
    2. Generate a maximum of 3 visualizations, including a correlation heatmap. Do not use "Country Names".
    3. Generate plots mainly for highly correlated fields.
    Exclude pairplots and focus on concise, meaningful plots.
    Export plots in PNG format as '1.png', '2.png', etc.
    Make sure there is only 1 plot per Image.
    The Image must be clear.
    Output only the Python code.
    """
    response = call_openai_api(api_details, "chat/completions", prompt)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("python").strip()
    print(clean_code)
    execute_code(clean_code)

def narrate_data(dataset, columns, datatypes, api_details):

    summary = dataset.describe()
    prompt = f"""
Create a README.md file for my data analysis project.
Name of the columns = {columns}
Datatype of columns = {datatypes}
Summary of dataset = {summary}

The README must include the following sections:
1. About the Data: Briefly describe the data, its source, key attributes, and unique characteristics.
2. Analysis Performed: List and describe the key analysis techniques, methods, and statistical approaches used.
3. Insights Discovered: Highlight the main takeaways, trends, and patterns revealed from the analysis.
4. Implications & Recommendations: Based on the insights, suggest actions, recommendations, or next steps.

Additionally, incorporate the following three image files as part of the README:

1.png: Include this image in the "Analysis Performed" section to visualize key analysis steps.
2.png: Place this image in the "Insights Discovered" section to illustrate key trends.
3.png: Place this image in the "Implications & Recommendations" section to support the conclusions.
Use proper Markdown syntax, including headers (##), bullet points, and image tags like ![Alt text](path/to/image.png).
The README should be clear, professional, and concise, following best practices for open-source documentation.
Do not include any explanations, justifications, or comments. Only output the Markdown file.
"""
    response = call_openai_api(api_details, "chat/completions", prompt, use_vision=True)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("markdown").strip()
    write_into_file(clean_code)

def main():
    """Main function to handle dataset processing."""
    dataset_file = sys.argv[1]
    encoding = detect_file_encoding(dataset_file)
    global dataset 
    dataset = load_dataset(dataset_file, encoding)
    api_details = initialize_api_details()
    # Handle null values
    handle_null_values(dataset)
    # Feature Relevance
    if dataset.shape[1] > 10:
        feature_relevance(dataset, api_details)

    le = LabelEncoder()
    for col in dataset.columns:
        if dataset[col].dtype == "object":
            dataset[col] = le.fit_transform(dataset[col])

    correlation_matrix = dataset.corr()

    # Handle outliers
    numerical_columns = [col for col in dataset.select_dtypes(include=['float64', 'int64']).columns]
    cleaned_data = handle_outliers(dataset, numerical_columns, api_details)
    # Visualize data
    columns = dataset.columns.tolist()
    datatypes = dataset.dtypes.tolist()
    visualize_data(cleaned_data, columns, datatypes, api_details, correlation_matrix)
    narrate_data(cleaned_data, columns, datatypes, api_details)

if __name__ == "__main__":
    main()
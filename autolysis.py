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
#     "tenacity>=9.0.0",
# ]
# ///

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import requests
import json
import chardet
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import base64
from tenacity import retry, stop_after_attempt, wait_fixed

def detect_file_encoding(file_path):
    """
    Detect the encoding of a given file.
    @param file_path - The path to the file whose encoding needs to be detected.
    @return The encoding of the file.
    """
    
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
    return result['encoding']

def load_dataset(file_path, encoding):
    """
    Load a dataset from a CSV file using the specified encoding.
    @param file_path: The path to the CSV file.
    @param encoding: The encoding to be used for reading the file.
    @return: The dataset loaded from the CSV file.
    """

    return pd.read_csv(file_path, encoding=encoding)

def initialize_api_details():
    """
    Initialize API details by loading environment variables and returning a dictionary containing the API token and base URL.
    @return A dictionary containing the API token and base URL.
    """

    load_dotenv()
    return {
        "token": os.getenv("AIPROXY_TOKEN"),
        "base_url": "https://aiproxy.sanand.workers.dev/openai/v1",
    }

def prepare_feature_relevance_prompt(dataset):
    """
    Prepare a prompt for an expert in data analysis to identify and remove irrelevant features from a dataset based on logical reasoning and analysis.
    @param dataset - the dataset to be analyzed
    @return A detailed prompt with information about the dataset, instructions for feature relevance analysis, and guidelines for feature removal.
    """

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
    """
    Encode an image file into base64 format.
    @param image_path - The path to the image file to be encoded.
    @raises FileNotFoundError if the image file is not found.
    @return The base64 encoded image as a string.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai_api(api_details, endpoint, prompt, functions=None, use_vision = False, readme = False):
    """
    Make a request to the OpenAI API and return the response.
    @param api_details - Details needed to access the API (base_url, token)
    @param endpoint - The specific endpoint to call in the API
    @param prompt - The input prompt for the API
    @param functions - Optional functions to include in the request
    @param use_vision - Flag indicating whether to use vision analysis
    @param readme - Flag indicating whether to integrate insights into the README
    @return The response from the API
    """

    url = f"{api_details['base_url']}/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_details['token']}"
    }
    if use_vision and readme:
        image_paths = ["1.png", "2.png", "3.png", "pairplot.png"]
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
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['pairplot.png']}"
                    }
                }
            ]
        }
    ]
}
    elif use_vision and not readme:
        image_paths = ['pairplot.png']
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
                        "url": f"data:image/png;base64,{encoded_images['pairplot.png']}"
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
    """
    Execute the Python code returned by the API.
    @param code - The Python code to be executed
    @return None
    """

    try:
        exec(code, globals())
        print("Code Executed Successfully")
    except Exception as e:
        raise Exception
        print("Error during code execution:", e)

def write_into_file(content):
    """
    Write the provided content into the README.md file.
    @param content - The content to be written into the file.
    """

    with open("README.md", "w") as file:
        file.write(content)

def handle_null_values(dataset, threshold=0.5, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """
    Handle null values in a dataset using a fixed heuristic for numeric and categorical variables.
    @param dataset - The dataset containing null values to be handled.
    @param threshold - The threshold for null ratios to drop columns (default is 0.5).
    @param numeric_strategy - The strategy to impute missing numeric values (default is 'mean').
    @param categorical_strategy - The strategy to impute missing categorical values (default is 'most_frequent').
    """

    null_ratios = dataset.isnull().mean()

    columns_to_drop = null_ratios[null_ratios > threshold].index
    dataset = dataset.drop(columns=columns_to_drop)

    numeric_cols = dataset.select_dtypes(include=['number']).columns
    categorical_cols = dataset.select_dtypes(exclude=['number']).columns

    if numeric_cols.any():
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

    if categorical_cols.any():
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

def feature_relevance(dataset, api_details):
    """
    This function dynamically adjusts the relevance of features in a dataset using statistical measures. It prepares a prompt for feature relevance, calls an OpenAI API with provided details, retrieves and cleans the generated code, and then executes the cleaned code.
    @param dataset - The dataset containing features.
    @param api_details - Details required to call the OpenAI API.
    @return None
    """

    prompt = prepare_feature_relevance_prompt(dataset)
    response = call_openai_api(api_details, "chat/completions", prompt)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("python").strip()
    print(clean_code)
    execute_code(clean_code)

def generate_boxplots(columns, dataset):
    """
    Generate box plots for the specified columns in the dataset to visualize the distribution and detect outliers.
    @param columns - The columns in the dataset for which box plots will be generated.
    @param dataset - The dataset containing the data to be visualized.
    @return None
    """

    plt.figure(figsize=(10, 6))
    dataset[columns].boxplot()
    plt.title('Box Plots for Outlier Detection')
    plt.show()

def remove_outliers(dataset, columns):
    """
    Remove outliers from a dataset based on specified columns using the IQR method.
    @param dataset - The dataset containing the data
    @param columns - The columns in the dataset to check for outliers
    @return A cleaned dataset with outliers removed
    """

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
    """
    Handle outliers in a dataset by generating box plots for outlier detection and removing identified outliers.
    @param dataset - The dataset containing the data
    @param columns - The numerical columns in the dataset
    @param api_details - Details required to call the OpenAI API
    @return cleaned_data - The dataset with outliers removed
    """
    
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

def generate_pairplot(dataset):
    """
    Generate a pairplot for the given dataset and save it as a PNG file.
    @param dataset - The dataset for which the pairplot will be generated.
    """

    sns.pairplot(data = dataset)
    plt.savefig("pairplot.png")

def visualize_data(dataset, columns, datatypes, api_details, correlation_matrix):
    """
    Generate visualizations for the dataset using Seaborn and Matplotlib based on the provided information.
    @param dataset - The dataset for which visualizations are to be generated.
    @param columns - List of column names in the dataset.
    @param datatypes - List of datatypes corresponding to the columns.
    @param api_details - Details required to call the OpenAI API for code completion.
    @param correlation_matrix - The correlation matrix between columns in the dataset.
    @return None. The function generates Python code for visualizing the dataset using Seaborn and Matplotlib.
    """

    prompt = f"""
    You are a Python Data Analyst.
    Given the following column names: {columns}
    And their respective datatypes: {datatypes}
    The Correlation between each Column: {correlation_matrix}
    The pairplot between the features is also given.

    Assume that the dataset is already imported.
    *IMPORTANT* Name of the dataset: "dataset"

    Generate Python code using Seaborn to:
    1. Convert any date column into datetime using "to_datetime" method in pandas.
    2. Generate a maximum of 3 visualizations, including a correlation heatmap. Do not use "Country Names".
    3. Generate plots mainly for highly correlated fields, using the Pairplot and correlation matrix.
    Exclude pairplots and focus on concise, meaningful plots.
    Export plots in PNG format as '1.png', '2.png', etc.
    Make sure there is only 1 plot per Image.
    The Image must be clear.
    Output only the Python code.
    """
    response = call_openai_api(api_details, "chat/completions", prompt, use_vision=True)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("python").strip()
    print(clean_code)
    execute_code(clean_code)

def narrate_data(dataset, columns, datatypes, api_details):
    """
    Narrate the data analysis project by creating a README.md file with specific sections and incorporating images.
    @param dataset - The dataset being analyzed.
    @param columns - Names of the columns in the dataset.
    @param datatypes - Datatypes of the columns in the dataset.
    @param api_details - Details required for the OpenAI API.
    @return None
    """

    summary = dataset.describe()
    prompt = f"""
Create a README.md file for my data analysis project.
Name of the columns = {columns}
Datatype of columns = {datatypes}
Summary of dataset = {summary}

Make use of the pairplot and images fed into to delve deeper into the insights.

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
    response = call_openai_api(api_details, "chat/completions", prompt, use_vision=True, readme=True)
    code = response['choices'][0]['message']['content']
    clean_code = code.strip("```").strip("markdown").strip()
    write_into_file(clean_code)

def encode(dataset):
    """
    Encode categorical variables in a dataset using LabelEncoder.
    @param dataset - The dataset containing categorical variables to be encoded.
    @return The dataset with encoded categorical variables.
    """
    
    le = LabelEncoder()
    for col in dataset.columns:
        if dataset[col].dtype == "object":
            dataset[col] = le.fit_transform(dataset[col])
    return dataset

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def main():
    """
    The main function that processes the dataset, including handling null values, feature relevance, encoding, outlier handling, visualization, and narration.
    @return None
    """

    dataset_file = sys.argv[1]
    encoding = detect_file_encoding(dataset_file)
    # Making the dataset global ensures that no errors and confusion occurs
    global dataset 
    dataset = load_dataset(dataset_file, encoding)
    api_details = initialize_api_details()
    # Handle null values
    handle_null_values(dataset)
    # Feature Relevance
    if dataset.shape[1] > 10:
        feature_relevance(dataset, api_details)
    dataset = encode(dataset)
    correlation_matrix = dataset.corr()
    # Handle outliers
    numerical_columns = [col for col in dataset.select_dtypes(include=['float64', 'int64']).columns]
    cleaned_data = handle_outliers(dataset, numerical_columns, api_details)
    # Visualize data
    columns = dataset.columns.tolist()
    datatypes = dataset.dtypes.tolist()
    generate_pairplot(dataset)
    visualize_data(cleaned_data, columns, datatypes, api_details, correlation_matrix)
    narrate_data(cleaned_data, columns, datatypes, api_details)

if __name__ == "__main__":
    """
    Check if the script is being run as the main program, and if so, call the main function.
    If the script is imported as a module, the main function will not be executed.
    This structure allows the script to be both imported and run directly.
    """
    main()
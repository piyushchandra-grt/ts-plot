import pandas as pd

def analyze_rhlf_data(input_filepath, output_filepath):
    """
    Analyzes RLHF data to count the number of '5' ratings for each model's
    completeness, correctness, and relevance.

    The data is grouped by 'Code Language', 'Industry', and 'Topic' to cover
    all entries accurately.

    Args:
        input_filepath (str): The path to the input CSV file.
        output_filepath (str): The path where the analyzed CSV will be saved.
    """
    try:
        # Read the source CSV file into a pandas DataFrame
        df = pd.read_csv(input_filepath)
        print("Successfully loaded the CSV file.")

        # Define all the rating columns that need to be analyzed
        rating_columns = [
            'Chat GPT o4-mini-high Human Completeness',
            'Chat GPT o4-mini-high Human Correctness',
            'Chat GPT o4-mini-high Human Relevance',
            'Gemini 2.5 pro Human Completeness',
            'Gemini 2.5 pro Human Correctness',
            'Gemini 2.5 pro Human Relevance',
            'Claude Opus 4 Human Completeness',
            'Claude Opus 4 Human Correctness',
            'Claude Opus 4 Human Relevance'
        ]

        # Ensure the rating columns are treated as numbers, not text.
        # Invalid values (like empty cells) will become 'NaN' and won't be counted.
        for col in rating_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # This is the key step: Group by all three columns.
        # This creates a unique group for "Python, Machine Learning, Time-Series Anomaly Detection",
        # another for "Cpp, Gaming, Simple 2D Physics Engine", etc. for ALL entries.
        grouped_data = df.groupby(['Code Language', 'Industry', 'Topic', 'Prompt ID'])

        # Create a list to hold the results before creating the final DataFrame
        results = []

        # Loop through each unique group to perform the analysis
        for name, group in grouped_data:
            # 'name' is a tuple like ('Python', 'Machine Learning', ...)
            # 'group' is a DataFrame containing all rows for that unique combination

            result_row = {
                'Code Language': name[0],
                'Industry': name[1],
                'Topic': name[2],
                'Prompt ID': name[3]
            }

            # For each rating column, count how many times the value is 5
            result_row['o4-mini-high - Completeness (count of 5s)'] = (group['Chat GPT o4-mini-high Human Completeness'] == 5).sum()
            result_row['o4-mini-high - Correctness (count of 5s)'] = (group['Chat GPT o4-mini-high Human Correctness'] == 5).sum()
            result_row['o4-mini-high - Relevance (count of 5s)'] = (group['Chat GPT o4-mini-high Human Relevance'] == 5).sum()

            result_row['Gemini 2.5 Pro - Completeness (count of 5s)'] = (group['Gemini 2.5 pro Human Completeness'] == 5).sum()
            result_row['Gemini 2.5 Pro - Correctness (count of 5s)'] = (group['Gemini 2.5 pro Human Correctness'] == 5).sum()
            result_row['Gemini 2.5 Pro - Relevance (count of 5s)'] = (group['Gemini 2.5 pro Human Relevance'] == 5).sum()

            result_row['Claude Opus 4 - Completeness (count of 5s)'] = (group['Claude Opus 4 Human Completeness'] == 5).sum()
            result_row['Claude Opus 4 - Correctness (count of 5s)'] = (group['Claude Opus 4 Human Correctness'] == 5).sum()
            result_row['Claude Opus 4 - Relevance (count of 5s)'] = (group['Claude Opus 4 Human Relevance'] == 5).sum()

            results.append(result_row)

        # Create the final DataFrame from our list of results
        final_df = pd.DataFrame(results)

        # Save the aggregated results to a new CSV file
        final_df.to_csv(output_filepath, index=False)

        print(f"\nAnalysis complete!")
        print(f"The aggregated data has been saved to: '{output_filepath}'")

    except FileNotFoundError:
        print(f"ERROR: The input file was not found. Please ensure '{input_filepath}' is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution ---A
if __name__ == "__main__":
    input_file = 'RLHF data - Main RLHF (4).csv'
    output_file = 'RLHF_data_analyzed_by_topic.csv'
    analyze_rhlf_data(input_file, output_file)
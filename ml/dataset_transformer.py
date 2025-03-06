import pandas as pd
import numpy as np

# NOTE: names of preset cols may be different based on dataset, this is just a generalized pipeline

CHOSEN_COLUMN = 'chosen'  # name of col with chosen responses
REJECTED_COLUMN = 'rejected'  # name of col with rejected responses
COLUMNS_TO_DROP = ['metadata', 'timestamp', 'id']  # cols to remove

def transform_rlhf_dataset(df, chosen_col=CHOSEN_COLUMN, rejected_col=REJECTED_COLUMN, drop_cols=COLUMNS_TO_DROP):
    """    
    Parameters:
    df (pandas.DataFrame): Input dataframe with chosen and rejected columns
    chosen_col (str): Name of column containing chosen responses
    rejected_col (str): Name of column containing rejected responses
    drop_cols (list): List of column names to drop from the dataset
    
    Returns:
    pandas.DataFrame: Transformed dataset with 'text' and 'label' columns
    """

    df = df.copy()

    existing_cols_to_drop = [col for col in drop_cols if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    preserved_cols = [col for col in df.columns if col not in [chosen_col, rejected_col]]
    
    # two separate dataframes for liked and disliked
    liked_df = df[[chosen_col]].copy()
    liked_df.columns = ['text']
    liked_df['label'] = 'liked'
    
    disliked_df = df[[rejected_col]].copy()
    disliked_df.columns = ['text']
    disliked_df['label'] = 'disliked'

    for col in preserved_cols:
        liked_df[col] = df[col]
    for col in preserved_cols:
        disliked_df[col] = df[col]
    
    # combine + shuffle
    transformed_df = pd.concat([liked_df, disliked_df], ignore_index=True)
    transformed_df = transformed_df.dropna(subset=['text'])
    transformed_df = transformed_df.sample(frac=1).reset_index(drop=True)

    # reordering
    column_order = ['text', 'label'] + preserved_cols
    transformed_df = transformed_df[column_order]
    
    return transformed_df

def test_example():
    example_data = {
        'chosen': ['This is a good response', 'Another good one'],
        'rejected': ['This is a bad response', 'Another bad one'],
        'metadata': ['meta1', 'meta2'],
        'timestamp': ['2024-01-01', '2024-01-02'],
        'id': [1, 2]
    }
    
    df = pd.DataFrame(example_data)
    transformed_df = transform_rlhf_dataset(
        df,
        chosen_col='chosen',
        rejected_col='rejected',
        drop_cols=['metadata', 'id']
    )
    
    print("Original shape:", df.shape)
    print("\nTransformed shape:", transformed_df.shape)
    print("\nTransformation sample:")
    print(transformed_df.head())
    print("\nLabel distribution:")
    print(transformed_df['label'].value_counts())

if __name__ == "__main__":
    test_example()
"""
Data cleaning and noise reduction module.
Normalizes text, strips leading mentions/handles, fixes HTML entities, and filters low-quality inputs.
"""

import re
import html
import pandas as pd

def clean_tweet_text(text: str, remove_mentions: bool = True) -> str:
    """
    Cleans raw tweet text.
    - Decodes HTML entities (&amp; -> &, etc.)
    - Optionally removes leading @user mentions (keeps main message text)
    - Replaces URLs with [URL] token
    - Collapses extra whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML
    cleaned = html.unescape(text)

    # Standardize URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', '[URL]', cleaned)

    if remove_mentions:
        # Strip leading @mentions
        cleaned = re.sub(r'^(?:@\w+\s*)+', '', cleaned)
        # Clean remaining inline mentions if appropriate or leave them sanitized
        cleaned = re.sub(r'@\w+', '[USER]', cleaned)

    # Collapse multiple whitespaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def clean_dataframe(df: pd.DataFrame, text_col: str = "text", min_len: int = 5) -> pd.DataFrame:
    """
    Applies cleaning across DataFrame and filters out empty or ultra-short messages.
    """
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_tweet_text)
    
    # Filter rows where cleaned text is shorter than min_len
    valid_mask = df["clean_text"].str.len() >= min_len
    cleaned_df = df[valid_mask].reset_index(drop=True)
    return cleaned_df

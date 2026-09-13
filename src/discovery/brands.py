"""
Brand analysis and evidence-based brand selection module.
Calculates brand statistics (tweet count, reconstructed conversation count, customer message count,
support response count, average turn length, resolution rate, noise score, intent diversity).
"""

import pandas as pd
from typing import Dict, List, Any
import numpy as np
from src.data.clean import clean_dataframe
from src.data.conversations import reconstruct_conversations

def analyze_all_brands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes all candidate brands present in the dataset and computes key metrics.
    """
    cleaned_df = clean_dataframe(df)
    
    # Identify brand handles (non-numeric author_ids)
    all_authors = cleaned_df["author_id"].dropna().unique()
    brand_candidates = [a for a in all_authors if not a.isdigit() and not a.startswith("cust_") and len(a) > 1]

    # Reconstruct all conversations
    all_convs = reconstruct_conversations(cleaned_df)
    convs_df = pd.DataFrame([c.model_dump() for c in all_convs]) if all_convs else pd.DataFrame()

    results = []

    for brand in brand_candidates:
        brand_tweets = cleaned_df[cleaned_df["author_id"] == brand]
        support_response_count = len(brand_tweets)
        
        if convs_df.empty:
            brand_convs = pd.DataFrame()
        else:
            brand_convs = convs_df[convs_df["brand"] == brand]
            
        conv_count = len(brand_convs)
        
        if conv_count == 0:
            continue

        customer_msg_count = conv_count  # paired customer queries
        avg_cust_len = brand_convs["customer_message"].str.len().mean() if conv_count > 0 else 0
        avg_resp_len = brand_convs["support_response"].str.len().mean() if conv_count > 0 else 0
        
        # Unique customer query ratio (measure of intent diversity / non-duplication)
        unique_cust_msgs = brand_convs["customer_message"].nunique() if conv_count > 0 else 0
        diversity_score = (unique_cust_msgs / conv_count) if conv_count > 0 else 0.0

        # Duplicate/noise score (percentage of exact repeated support responses e.g. canned responses)
        canned_response_ratio = 1.0 - (brand_convs["support_response"].nunique() / conv_count) if conv_count > 0 else 0.0

        # Overall quality score (balances volume, conversation depth, and diversity)
        quality_score = conv_count * (1.0 - canned_response_ratio) * diversity_score

        results.append({
            "brand": brand,
            "total_support_tweets": support_response_count,
            "reconstructed_conversations": conv_count,
            "customer_message_count": customer_msg_count,
            "avg_cust_length_chars": round(avg_cust_len, 1),
            "avg_resp_length_chars": round(avg_resp_len, 1),
            "unique_customer_queries": unique_cust_msgs,
            "intent_diversity_ratio": round(diversity_score, 3),
            "canned_response_ratio": round(canned_response_ratio, 3),
            "quality_score": round(quality_score, 1)
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by="reconstructed_conversations", ascending=False).reset_index(drop=True)
        
    return result_df

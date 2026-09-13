"""
Conversation thread reconstruction module.
Links customer queries with brand support responses using in_reply_to_tweet_id and response_tweet_id tree traversals.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ConversationPair(BaseModel):
    conversation_id: str
    brand: str
    customer_id: str
    customer_message: str
    raw_customer_text: str
    customer_created_at: str
    support_response: str
    raw_support_text: str
    support_created_at: str
    turn_count: int = 2


def reconstruct_conversations(df: pd.DataFrame, target_brand: Optional[str] = None) -> List[ConversationPair]:
    """
    Reconstructs customer query -> brand response pairs from the Twitter dataset.
    If target_brand is specified, filters conversations for that specific brand.
    """
    # Create lookup map by tweet_id
    tweet_map = {}
    for idx, row in df.iterrows():
        t_id = str(row["tweet_id"]).strip()
        if t_id and t_id != "nan":
            tweet_map[t_id] = row.to_dict()

    conversations: List[ConversationPair] = []

    for idx, row in df.iterrows():
        author = str(row["author_id"]).strip()
        in_reply_to = str(row["in_reply_to_tweet_id"]).strip()
        response_id_str = str(row["response_tweet_id"]).strip()

        # Identify customer tweets (author is not a brand handle like AppleSupport, AmazonHelp, etc.)
        # In TWCS, brand tweets are usually handles (letters/underscores) while customers are numeric IDs (or non-brand)
        is_brand_author = not author.isdigit() and len(author) > 1 and not author.startswith("cust_")
        
        if is_brand_author:
            # If target_brand filter is active, check brand name
            if target_brand and author.lower() != target_brand.lower():
                continue

            # This is a brand tweet. Check if it replies to a customer tweet
            parent_tweet = tweet_map.get(in_reply_to)
            if parent_tweet:
                cust_author = str(parent_tweet.get("author_id", "")).strip()
                # Ensure parent is a customer tweet
                if cust_author and (cust_author.isdigit() or cust_author.startswith("cust_")):
                    cust_raw = str(parent_tweet.get("text", ""))
                    brand_raw = str(row.get("text", ""))
                    
                    cust_clean = parent_tweet.get("clean_text", cust_raw)
                    brand_clean = row.get("clean_text", brand_raw)

                    conv = ConversationPair(
                        conversation_id=f"conv_{in_reply_to}_{row['tweet_id']}",
                        brand=author,
                        customer_id=cust_author,
                        customer_message=cust_clean,
                        raw_customer_text=cust_raw,
                        customer_created_at=str(parent_tweet.get("created_at", "")),
                        support_response=brand_clean,
                        raw_support_text=brand_raw,
                        support_created_at=str(row.get("created_at", "")),
                        turn_count=2
                    )
                    conversations.append(conv)

    logger.info(f"Reconstructed {len(conversations)} conversation pairs.")
    return conversations


def conversations_to_dataframe(conversations: List[ConversationPair]) -> pd.DataFrame:
    """Converts a list of ConversationPair models to a pandas DataFrame."""
    return pd.DataFrame([c.model_dump() for c in conversations])

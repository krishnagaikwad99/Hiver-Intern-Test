"""
Unit tests for data loading, cleaning, and conversation reconstruction modules.
"""

import unittest
import pandas as pd
from src.data.clean import clean_tweet_text, clean_dataframe
from src.data.conversations import reconstruct_conversations

class TestDataModule(unittest.TestCase):
    def test_clean_tweet_text(self):
        raw = "@AppleSupport My screen is broken &amp; freezing! Check https://apple.com"
        cleaned = clean_tweet_text(raw, remove_mentions=True)
        self.assertNotIn("@AppleSupport", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("&", cleaned)
        self.assertIn("[URL]", cleaned)
        self.assertIn("screen is broken", cleaned)

    def test_clean_dataframe(self):
        df = pd.DataFrame([
            {"tweet_id": "1", "text": "@AppleSupport Help me please!"},
            {"tweet_id": "2", "text": "hi"}
        ])
        cleaned_df = clean_dataframe(df, min_len=5)
        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(cleaned_df.iloc[0]["tweet_id"], "1")

    def test_reconstruct_conversations(self):
        df = pd.DataFrame([
            {
                "tweet_id": "101",
                "author_id": "cust_123",
                "in_reply_to_tweet_id": "",
                "created_at": "Wed Oct 11 11:53:00 +0000 2017",
                "text": "@AppleSupport My battery drops fast",
                "response_tweet_id": "102",
                "in_response_to_tweet_id": ""
            },
            {
                "tweet_id": "102",
                "author_id": "AppleSupport",
                "in_reply_to_tweet_id": "101",
                "created_at": "Wed Oct 11 11:55:00 +0000 2017",
                "text": "@cust_123 Try restarting your device",
                "response_tweet_id": "",
                "in_response_to_tweet_id": "101"
            }
        ])
        convs = reconstruct_conversations(df, target_brand="AppleSupport")
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].brand, "AppleSupport")
        self.assertEqual(convs[0].customer_id, "cust_123")
        self.assertIn("battery drops fast", convs[0].customer_message)
        self.assertIn("restarting your device", convs[0].support_response)

if __name__ == "__main__":
    unittest.main()

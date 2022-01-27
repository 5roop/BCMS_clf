import os
import parse
import logging
import pandas as pd
from typing import Set, List
from transliterate import translit

LABELS = ["hr", "bs", "sr", "me"]
#%%
def transliterate(input_text: str) -> str:
    """Transliterates text from Serbian cyrillic script to latin.

    Args:
        input_text (str): Possibly cyrillic or latin text.

    Returns:
        str: Latin transcription.
    """    
    from transliterate import translit
    return translit(input_text, "sr", reversed=True)

def is_alpha(token: str) -> bool:
    import re
    pattern = "^[a-zšđčćž]+$"
    compiled_pattern = re.compile(pattern)
    return bool(compiled_pattern.match(token))

def load_SET_dataset() -> pd.DataFrame:
    """Reads SETimes dataset from web.

    Returns:
        pd.DataFrame: dataframe with the dataset.
    """    
    url = "https://github.com/5roop/task4/raw/main/setimes_dataset/SETimes.json"

    return pd.read_json(url)

def load_twitter_dataset() -> pd.DataFrame:
    """Reads twitter dataset from web.

    Returns:
        pd.DataFrame: dataframe with the dataset.
    """  
    url = "https://github.com/5roop/task4/raw/main/twitter_dataset/twitter.json"
    
    return pd.read_json(url)
# %%

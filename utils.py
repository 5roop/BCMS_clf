#%% 
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
    """Checks if the input string is strictly lowercase without numerals.

    Args:
        token (str): Input text.

    Returns:
        bool: Result of checking.
    """    
    import re
    pattern = "^[a-zšđčćž]+$"
    compiled_pattern = re.compile(pattern)
    return bool(compiled_pattern.match(token))

def load_SET_dataset():
    """Reads SETimes dataset from web.

    Returns:
        pd.DataFrame: dataframe with the dataset.
    """    
    url = "https://github.com/5roop/task4/raw/main/setimes_dataset/SETimes.json"

    return pd.read_json(url)

def load_twitter_dataset():
    """Reads twitter dataset from web.

    Returns:
        pd.DataFrame: dataframe with the dataset.
    """  
    url = "https://github.com/5roop/task4/raw/main/twitter_dataset/twitter.json"
    
    return pd.read_json(url)


def get_N_tokens(N=5000) -> Set[str]:
    """Get N most important tokens per every language pair.
    Reads tokens from repo, calculates Keyness scores, and
    returns union of sets of N most important tokens

    Args:
        N (int, optional): How many most unique tokens per language to take. 
                           Defaults to 5000.

    Returns:
        Set[str]: Resulting set of tokens.
    """    
    import pandas as pd
    import numpy as np
    url = "https://github.com/5roop/task4/raw/main/toy_tokens.csv"
    df = pd.read_csv(url, index_col=0)

    df = df.iloc[~df.index.isna(), :]

    def filter_token(token: str) -> bool:
        token = token.replace(" ", "")
        if len(token) < 3:
            return False
        return any([vowel in token for vowel in "aeiou"])
    df["keep"] = df.index.copy()
    df["keep"] = df.keep.apply(filter_token)
    df = df.loc[df.keep, :]
    df.drop(columns=["keep"], inplace=True)
    NUM_FEATS = N
    for column in df.columns:
        new_column_name = column + "_f"
        corpus_size = df[column].sum()
        df[new_column_name] = df[column] * 1e6 / corpus_size

    N = 1

    df["HR_SR"] = (df["hrwac_head_pp_f"] + N) / (df["srwac_head_pp_f"] + N)
    df["SR_HR"] = (df["srwac_head_pp_f"] + N) / (df["hrwac_head_pp_f"] + N)

    df["HR_CNR"] = (df["hrwac_head_pp_f"] + N) / (df["cnrwac_head_pp_f"] + N)
    df["CNR_HR"] = (df["cnrwac_head_pp_f"] + N) / (df["hrwac_head_pp_f"] + N)

    df["HR_BS"] = (df["hrwac_head_pp_f"] + N) / (df["bswac_head_pp_f"] + N)
    df["BS_HR"] = (df["bswac_head_pp_f"] + N) / (df["hrwac_head_pp_f"] + N)

    df["BS_SR"] = (df["bswac_head_pp_f"] + N) / (df["srwac_head_pp_f"] + N)
    df["SR_BS"] = (df["srwac_head_pp_f"] + N) / (df["bswac_head_pp_f"] + N)

    df["BS_CNR"] = (df["bswac_head_pp_f"] + N) / (df["cnrwac_head_pp_f"] + N)
    df["CNR_BS"] = (df["cnrwac_head_pp_f"] + N) / (df["bswac_head_pp_f"] + N)

    df["CNR_SR"] = (df["cnrwac_head_pp_f"] + N) / (df["srwac_head_pp_f"] + N)
    df["SR_CNR"] = (df["srwac_head_pp_f"] + N) / (df["cnrwac_head_pp_f"] + N)

    combos = ['HR_SR', 'SR_HR', 'HR_CNR', 'CNR_HR', 'HR_BS', 'BS_HR',
              'BS_SR', 'SR_BS', 'BS_CNR', 'CNR_BS', 'CNR_SR', 'SR_CNR']
    important_features = set()
    for lang_comb in combos:
        s = df[lang_comb].sort_values(ascending=False)
        current_features = s.index[:NUM_FEATS].values
        important_features = important_features.union(set(current_features))
    try:
        important_features.remove(np.nan)
    except KeyError:
        pass
    return important_features


# %%

import numpy as np
import pandas as pd
import random
import re
import psycopg2 as ps
import string
import emoji
import joblib

from string import digits

def is_Singlish(word_test):
    def extract_emojis(s):
        return ' '.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

    def remove_emojis(text: str) -> str:
        return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI['en'])

    emoji_list= extract_emojis(word_test)
    hate_list=check_Hate_emoji(emoji_list)
    offencive_list = check_Offencive_emoji(emoji_list)
    normal_list = check_Normal_emoji(emoji_list)
    try:
        remove_emojis(word_test).encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False,hate_list,offencive_list,normal_list
    else:
        return True,hate_list,offencive_list,normal_list


def check_Hate_emoji(emoji_list):
    hatelist=["😡","💩","🤓","😃","😞","🤬","👉","😝","🔫","😜","🍆","🍌","👈","🔞","💕","😔","🐕","😷","🖕","🤧","🤭","🤢","🤮","😠"]
    return list(set(hatelist).intersection(emoji_list))

def check_Offencive_emoji(emoji_list):
    offencivelist=["🙉","😅","😆","✋","🤣","💔","😉","🌹","👌","😀","😊","💵","🤒","🤕","♂","💙","💛","💜","🤦","☺","💘","😒","😑","😛","😩","😶","😓","😂","😁"]
    return list(set(offencivelist).intersection(emoji_list))

def check_Normal_emoji(emoji_list):
    normallist=["🙏","🎀","","💙","💛","💜","💯","☺","😛","☸","✝","😍","😋","😢","🙌","🌷","🔥","❤","👊","👙","💐","🧐","💚","👍","😻","🛒","🖤","😎","💦","😌","♥","😪","🤗","😭","😘","😥"]
    return list(set(normallist).intersection(emoji_list))

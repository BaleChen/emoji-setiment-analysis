# reference: https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-using-word2vec-bilstm

import emoji
import re
from contraction_map import contraction_map as cm

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern = '@[^\s]+'
hashtagPattern = '#[^\s]+'
alphaPattern = "[^a-z0-9<>]"
sequencePattern = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

def is_alnum_or_emoji_or_space(char):
    return char.isalnum() or emoji.is_emoji(char) or char in ('\t', ' ')

def preprocess_apply(tweet):
    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern, '<url>', tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern, '<user>', tweet)

    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    for contraction, replacement in cm.CONTRACTION_MAP.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = ''.join(filter(is_alnum_or_emoji_or_space, tweet))

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet


# End of reference. The following code is wrote by me.

def emoji2description(text):
    return emoji.replace_emoji(text, replace=lambda chars, data_dict: ' '.join(data_dict['en'].split('_')).strip(':'))


def emoji2concat_description(text):
    emoji_list = emoji.emoji_list(text)
    ret = emoji.replace_emoji(text, replace='').strip()
    for json in emoji_list:
        this_desc = ' '.join(emoji.EMOJI_DATA[json['emoji']]['en'].split('_')).strip(':')
        ret += ' ' + this_desc
    return ret


def extract_emojis(text):
    emoji_list = emoji.emoji_list(text)
    #     print(emoji_list)
    ret = []
    for json in emoji_list:
        this_emoji = json['emoji']
        ret.append(this_emoji)
    return ' '.join(ret)


def keep_only_emojis(data):
    cnt = data['content'].apply(emoji.emoji_count)
    return data[cnt >= 1]

def remove_emojis(text):
    return re.sub('[^a-z0-9<>]', '', text)
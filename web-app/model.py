
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import app, Sort, reviews_all
import nltk
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import re
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from collections import Counter

def pair_split(x):

    stop = stopwords.words('english')
    stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
                'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']
    words = re.sub('[^A-Za-z_]+', ' ', x)
    words = words.split()
    words_new = [x for x in words if x not in stop]
    if len(words_new) == 1:
        return words_new
    else:
        pairs = [words_new[i]+'_'+words_new[i+1] for i in range(len(words_new)-1)]
        return pairs

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

def give_keytext(appid):
    fi_user_reviews = reviews_all(appid,sort=Sort.NEWEST)

    # convert the revies data into pandas dataframe
    df_user_reviews = pd.DataFrame(np.array(fi_user_reviews), columns=['review'])
    df_user_reviews = df_user_reviews.join(pd.DataFrame(df_user_reviews.pop('review').tolist()))

    # removing comments less than 4 words
    df_reviews_updated = df_user_reviews[(df_user_reviews['content'].str.count(' ') > 3 )]

    df = df_reviews_updated[['content', 'score']]

    # clean text data
    df_clean = df
    df_clean["content"] = df_clean["content"].apply(lambda x: clean_text(x))

    pos_review_lower = df_clean[df_clean['score'] > 3]['content'].str.lower().str.cat(sep=' ')
    neg_review_lower = df_clean[df_clean['score'] < 3]['content'].str.lower().str.cat(sep=' ')
    neu_review_lower = df_clean[df_clean['score'] == 3]['content'].str.lower().str.cat(sep=' ')

    ## Remove Punctuations

    pos_review_remove_pun = re.sub('[^A-Za-z]+', ' ', pos_review_lower)
    neg_review_remove_pun = re.sub('[^A-Za-z]+', ' ', neg_review_lower)
    neu_review_remove_pun = re.sub('[^A-Za-z]+', ' ', neu_review_lower)

    stop = stopwords.words('english')
    stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
                'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']

    #remove all the stopwords from the text
    pos_word_tokens_tags = word_tokenize(pos_review_remove_pun)
    neg_word_tokens_tags = word_tokenize(neg_review_remove_pun)
    neu_word_tokens_tags = word_tokenize(neu_review_remove_pun)
    pos_filtered_sentence_tags = [w_tags for w_tags in pos_word_tokens_tags if not w_tags in stop]
    pos_filtered_sentence_tags = []
    for w_tags in pos_word_tokens_tags:
        if w_tags not in stop:
            pos_filtered_sentence_tags.append(w_tags)

    neg_filtered_sentence_tags = [w_tags for w_tags in neg_word_tokens_tags if not w_tags in stop]
    neg_filtered_sentence_tags = []
    for w_tags in neg_word_tokens_tags:
        if w_tags not in stop:
            neg_filtered_sentence_tags.append(w_tags)
            
    neu_filtered_sentence_tags = [w_tags for w_tags in neu_word_tokens_tags if not w_tags in stop]
    neu_filtered_sentence_tags = []
    for w_tags in neu_word_tokens_tags:
        if w_tags not in stop:
            neu_filtered_sentence_tags.append(w_tags)


    pos_without_single_chr_rev = [word_tags for word_tags in pos_filtered_sentence_tags if len(word_tags) > 2]
    neg_without_single_chr_rev = [word_tags for word_tags in neg_filtered_sentence_tags if len(word_tags) > 2]
    neu_without_single_chr_rev = [word_tags for word_tags in neu_filtered_sentence_tags if len(word_tags) > 2]

    pos_review_lower = df_clean[df_clean['score'] > 3]['content'].str.lower().apply(pair_split).apply(lambda x: " ".join(x)).str.cat(sep=' ')
    neg_review_lower = df_clean[df_clean['score'] < 3]['content'].str.lower().apply(pair_split).apply(lambda x: " ".join(x)).str.cat(sep=' ')
    neu_review_lower = df_clean[df_clean['score'] == 3]['content'].str.lower().apply(pair_split).apply(lambda x: " ".join(x)).str.cat(sep=' ')

    pos_review_lower_rem = pos_review_lower.split(' ')
    pos_review_lower_rem = [a for a  in pos_review_lower_rem if a.find('_') >0]
    pos_review_remove_pun = " ".join(pos_review_lower_rem)

    neg_review_lower_rem = neg_review_lower.split(' ')
    neg_review_lower_rem = [a for a  in neg_review_lower_rem if a.find('_') >0]
    neg_review_remove_pun = " ".join(neg_review_lower_rem)

    neu_review_lower_rem = neu_review_lower.split(' ')
    neu_review_lower_rem = [a for a  in neu_review_lower_rem if a.find('_') >0]
    neu_review_remove_pun = " ".join(neu_review_lower_rem)

    pos_word_tokens_tags = word_tokenize(pos_review_remove_pun)
    neg_word_tokens_tags = word_tokenize(neg_review_remove_pun)
    neu_word_tokens_tags = word_tokenize(neu_review_remove_pun)
    pos_filtered_sentence_tags = [w_tags for w_tags in pos_word_tokens_tags if not w_tags in stop]
    pos_filtered_sentence_tags = []
    for w_tags in pos_word_tokens_tags:
        if w_tags not in stop:
            pos_filtered_sentence_tags.append(w_tags)

    neg_filtered_sentence_tags = [w_tags for w_tags in neg_word_tokens_tags if not w_tags in stop]
    neg_filtered_sentence_tags = []
    for w_tags in neg_word_tokens_tags:
        if w_tags not in stop:
            neg_filtered_sentence_tags.append(w_tags)
            
    neu_filtered_sentence_tags = [w_tags for w_tags in neu_word_tokens_tags if not w_tags in stop]
    neu_filtered_sentence_tags = []
    for w_tags in neu_word_tokens_tags:
        if w_tags not in stop:
            neu_filtered_sentence_tags.append(w_tags)

    # Remove characters which have length less than 2  

    pos_without_single_chr_rev = [word_tags for word_tags in pos_filtered_sentence_tags if len(word_tags) > 2]
    neg_without_single_chr_rev = [word_tags for word_tags in neg_filtered_sentence_tags if len(word_tags) > 2]
    neu_without_single_chr_rev = [word_tags for word_tags in neu_filtered_sentence_tags if len(word_tags) > 2]


    counts = Counter(neg_without_single_chr_rev)
    count_top30 = counts.most_common(10)

    count_top30_df = pd.DataFrame(count_top30, columns=["Phrases","Count"])
    # plt.figure(figsize=(10, 16))
    plt.figure()
    sns.set(font_scale=1)
    category_plot = sns.barplot(x="Phrases",y ="Count",data=count_top30_df, palette = "RdYlBu")
    category_plot.set_xticklabels(category_plot.get_xticklabels(), rotation=90, ha="right")
    plt.title('Top issues as per User Reviews',size = 18)

    import os
    my_path = os.path.abspath(__file__ + '/../')
    plt.savefig(my_path.replace('\\', '/') + '/static/images/Negative_Phrases.png', bbox_inches="tight")
    # print(my_path.replace('\\', '/'))
    return count_top30

# keyphrases = give_keytext('com.epifi.paisa')
# print(keyphrases)

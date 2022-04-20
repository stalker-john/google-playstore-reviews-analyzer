from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import app, Sort, reviews_all

apps = {
    'finance':'com.epifi.paisa',
    'fashion':'com.fsn.nds',
    'communication':'io.walkietalkie',
    'health':'com.healthkart.healthkart',
    'photography':'com.gurushots.app',
    'weather':'com.apalon.weatherlive.free',
    'sports':'com.dream11sportsguru',
    'games':'com.inspiredsquare.blocks', 
    'music':'com.esound',
    'health':'com.apollo.patientapp',
}

def get_data(category, app_id):

    print(category, app_id)
    user_reviews = reviews_all(app_id,sort=Sort.NEWEST)

    # convert the revies data into pandas dataframe
    df_user_reviews = pd.DataFrame(np.array(user_reviews), columns=['review'])
    df_user_reviews = df_user_reviews.join(pd.DataFrame(df_user_reviews.pop('review').tolist()))

    # save in csv file
    df_user_reviews.to_csv(category + '.csv')


for key, value in apps.items():
    get_data(key, value)
    break
import pandas as pd

COLS_TO_DROP = ['id', 'Body ID', 'Stance', 'Stance_categ']


def add_polarity_product(df):
    df['prod_compound'] = df['article_compound'] * df['headline_compound']
    df['prod_neu'] = df['article_neu'] * df['headline_neu']
    df['prod_neg'] = df['article_neg'] * df['headline_neg']
    df['prod_pos'] = df['article_pos'] * df['headline_pos']
    return df


def get_data(features=None):
    df = pd.read_csv('train_all_feats.csv')
    df = df.drop('Unnamed: 0', axis=1)
    test_df = pd.read_csv('test_all_feats.csv')
    test_df = test_df.drop('Unnamed: 0', axis=1)
    df = add_polarity_product(df)
    test_df = add_polarity_product(test_df)

    df['Stance_categ'] = df['Stance'].astype('category',
                                             categories=['agree', 'discuss', 'disagree', 'unrelated']).cat.codes
    test_df['Stance_categ'] = test_df['Stance'].astype('category', categories=['agree', 'discuss', 'disagree',
                                                                               'unrelated']).cat.codes
    X_df = df.drop(COLS_TO_DROP, axis=1)
    test_X_df = test_df.drop(COLS_TO_DROP, axis=1)

    if features is not None:
        X_df = X_df[features]
        test_X_df = test_X_df[features]

    X = X_df.as_matrix()
    test_X = test_X_df.as_matrix()

    y = df['Stance_categ'].as_matrix()
    test_y = test_df['Stance_categ'].as_matrix()
    return (X_df, X, y, test_X_df, test_X, test_y)

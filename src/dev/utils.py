# Global import
import re
from pandas import isna
from numpy import nan
from sklearn.preprocessing import StandardScaler


def prepare_titanic_data(df):
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Separate letter from number on cabin
    df['cabin_letter'] = df['cabin'].apply(
        lambda x: [re.sub(u'[^a-z]', '', c.lower()) for c in x.split()][0] if not isna(x) else None
    ).replace('', 'none')
    df['cabin_num'] = df['cabin'].apply(
        lambda x: max([int(re.sub(u'[^0-9]', '0', c)) for c in x.split()]) if not isna(x) else None
    )

    # Separate letter from number on ticket
    df['ticket_letter'] = df['ticket'].apply(
        lambda x: re.sub(u'[^A-Z]', '', x) if not isna(x) else None
    ).replace('', 'none')

    df['ticket_num'] = df['ticket'].apply(
        lambda x: re.sub(u'[^0-9]', '0', x.lower()) if not isna(x) else None
    ).astype(int)

    # Fill missing values
    df['cabin_letter'].fillna('unknown', inplace=True)
    df['cabin_num'] = df['cabin_num'].fillna(-1).astype(int)
    df['ticket_letter'].fillna('unknown', inplace=True)
    df['ticket_num'].fillna(-1, inplace=True)

    df['pclass'].fillna(-1, inplace=True)
    df['sex'].fillna('unknown', inplace=True)

    mean_age = df['age'].mean()
    df['age'].fillna(mean_age, inplace=True)

    mean_sibsp = int(df['sibsp'].mean())
    df['sibsp'].fillna(mean_sibsp, inplace=True)

    mean_parch = int(df['parch'].mean())
    df['parch'].fillna(mean_parch, inplace=True)

    mean_fare = df['fare'].mean()
    df['fare'].fillna(mean_fare, inplace=True)
    df['embarked'].fillna('unknown', inplace=True)
    df['age'].fillna(mean_age, inplace=True)
    df['age'].fillna(mean_age, inplace=True)
    df['age'].fillna(mean_age, inplace=True)

    return df


def prepare_higgs_data(df, l_col_cats, l_targets, missing_value=None, scaler=None):

    # get list of numerical features
    l_num_features = sorted([c for c in df.columns if c not in l_col_cats + l_targets])

    # Transform to nan missing value
    if missing_value is not None:
        df = df.replace(to_replace=missing_value, value=nan)

    # Create scaler
    if scaler is None:
        scaler = StandardScaler()
        ax_standardize = scaler.fit_transform(df[l_num_features])

    else:
        ax_standardize = scaler.transform(df[l_num_features])

    # Fit scaler and standardized numerical features
    df = df.assign(**{c: ax_standardize[:, i] for i, c in enumerate(l_num_features)})

    return df, scaler

def collate_rows(df, col, group_by):
    """ Collate multiple rows of a column into one row

    Parameters
    ----------
    df : pd.DataFrame

    col : str
        name of column that stores records to be collated

    group_by : str
        name of column to group by

    Returns
    -------
    collated_records : pd.Series

    Examples
    ------
    collate_rows(df, col='user_value', group_by='user_id')
    # return a ds indexed by 'user_id'
    """

    collated_records = df.groupby(group_by)[col] \
        .apply(lambda x: x.str.cat(sep=',', na_rep=''))
    return collated_records
"""Extract the 2017-11-30 cross-section from the full panel data_ml.csv,
producing data_ml_2017_11.csv.

Both this section's (5.5) and Section 6.4's experiments take this
cross-section as input. The repo already ships a pre-generated
``data/data_ml_2017_11.csv``, so this script usually doesn't need to be
rerun; it's only needed when switching to a cross-section from a
different date:

    python chap5/5.5/get_data.py
"""

import pandas as pd

from paths import DATA_DIR, data_file

#: The cross-section date to extract. Change this to generate the
#: cross-section for a different date.
CROSS_SECTION_DATE = '2017/11/30'


def get_data():
    """Return the CROSS_SECTION_DATE cross-section, indexed by stock_id."""
    data_raw = pd.read_csv(data_file('data_ml.csv'))
    idx_date = data_raw.index[(data_raw['date'] == CROSS_SECTION_DATE)].tolist()
    data = data_raw.iloc[idx_date]  # filtering the dataset according to date index
    data = pd.DataFrame(data)
    data = data.set_index('stock_id')
    data = data.iloc[:, 1:-3].dropna()
    return data


def main():
    data = get_data()
    output_path = DATA_DIR / "data_ml_2017_11.csv"
    data.to_csv(output_path)
    print(f"Generated {output_path}")
    print(data.shape)
    print(data.info())
    print(data.describe())
    print(data.iloc[:5, 1])


if __name__ == "__main__":
    main()

import os
from utils.audio_utils import get_signal_wav


class FREDFeature:

    def __init__(self, wav_path=None, dialect_code=None, dialect=None, dialect_label=None, county=None,
                 county_code=None, county_label=None):
        self._wav_path = wav_path
        self._dialect_code = dialect_code
        self._dialect = dialect
        self._dialect_label = dialect_label
        self._county = county
        self._county_code = county_code
        self._county_label = county_label
        self._signal, self._sample_rate = get_signal_wav(self.wav_path)
        if self.county_code is not None or self.county is not None or self.county_label is not None:
            self.generate_county_info()
        elif self.dialect_code is not None or self.dialect is not None or self.dialect_label is not None:
            self.generate_dialect_info()

    def generate_county_info(self):
        if self.county_code is not None:
            self._county = DialectMapping.code2county[self.county_code]
            self._county_label = DialectMapping.county2idx[self.county_code]
            self._dialect_code = DialectMapping.county2dialect[self.county_code]
        elif self.county is not None:
            self._county_code = DialectMapping.county2code[self.county]
            self._county_label = DialectMapping.county2idx[self.county_code]
            self._dialect_code = DialectMapping.county2dialect[self.county_code]
        elif self.county_label is not None:
            self._county_code = DialectMapping.idx2county[self.county_label]
            self._county = DialectMapping.code2county[self.county_code]
            self._dialect_code = DialectMapping.county2dialect[self.county_code]
        self.generate_dialect_info()

    def generate_dialect_info(self):

        if self.dialect_code is not None:
            self._dialect_label = DialectMapping.dialect2idx[self.dialect_code]
            self._dialect = DialectMapping.code2dialect[self.dialect_code]

        elif self.dialect_label is not None:
            self._dialect_code = DialectMapping.idx2dialect[self.dialect_label]
            self._dialect = DialectMapping.code2dialect[self.dialect_code]

        elif self.dialect is not None:
            self._dialect_code = DialectMapping.code2dialect[self.dialect]
            self._dialect_label = DialectMapping.dialect2idx[self.dialect_code]

    @property
    def wav_path(self):
        return self._wav_path

    @property
    def dialect_label(self):
        return self._dialect_label

    @property
    def dialect_code(self):
        return self._dialect_code

    @property
    def dialect(self):
        return self._dialect

    @property
    def county_label(self):
        return self._county_label

    @property
    def county_code(self):
        return self._county_code

    @property
    def county(self):
        return self._county


def get_files_labels(datadir, train=True):
    fred_objects = []

    # if train:
    #     datadir = os.path.join(datadir, 'train')
    # else:
    #     datadir = os.path.join(datadir, 'dev')
    for dialect in os.listdir(datadir):
        print(dialect)
        if not os.path.isdir(os.path.join(datadir, dialect)):
            continue
        for f in os.listdir(os.path.join(datadir, dialect)):
            if not f.endswith('.wav'):
                continue
            county_code = f.split('_')[0]
            fred_objects.append(FREDFeature(wav_path=os.path.join(datadir, dialect, f), county_code=county_code))
    return fred_objects


class DialectMapping:
    county2code = {
        'Cornwall': 'CON',
        'Devon': 'DEV',
        'Oxfordshire': 'OXF',
        'Somerset': 'SOM',
        'Wiltshire': 'WIL',
        'Kent': 'KEN',
        'London': 'LND',
        'Middlesex': 'MDX',
        'Leicestershire': 'LEI',
        'Nottinghamshire': 'NTT',
        'Durham': 'DUR',
        'Lancashire': 'LAN',
        'Northumberland': 'NBL',
        'Westmorland': 'WES',
        'Yorkshire': 'YKS',
        'East London': 'ELN',
        'Midlothian': 'MLN',
        'West Lothian': 'WLN'
    }

    code2county = {
        'CON': 'Cornwall',
        'DEV': 'Devon',
        'DUR': 'Durham',
        'ELN': 'East Lothian',
        'KEN': 'Kent',
        'LAN': 'Lancashire',
        'LEI': 'Leicestershire',
        'LND': 'London',
        'MDX': 'Middlesex',
        'MLN': 'Midlothian',
        'NBL': 'Northumberland',
        'NTT': 'Nottinghamshire',
        'OXF': 'Oxfordshire',
        'SOM': 'Somerset',
        'WES': 'Westmorland',
        'WIL': 'Wiltshire',
        'WLN': 'West Lothian',
        'YKS': 'Yorkshire'
    }

    county2dialect = {
        'CON': 'SW',
        'DEV': 'SW',
        'DUR': 'N',
        'ELN': 'SCL',
        'KEN': 'SE',
        'LAN': 'N',
        'LEI': 'MID',
        'LND': 'SE',
        'MDX': 'SE',
        'MLN': 'SCL',
        'NBL': 'N',
        'NTT': 'MID',
        'OXF': 'SW',
        'SOM': 'SW',
        'WES': 'N',
        'WIL': 'SW',
        'WLN': 'SCL',
        'YKS': 'N'
    }

    code2dialect = {
        'MID': 'Midlands',
        'N': 'North',
        'SCL': 'Scottish Lowlands',
        'SE': 'Southeast',
        'SW': 'Southwest'
    }

    dialect2idx = {
        'MID': 0,
        'N': 3,
        'SCL': 2,
        'SE': 1,
        'SW': 4
    }

    idx2dialect = {
        0: 'MID',
        1: 'SE',
        2: 'SCL',
        3: 'N',
        4: 'SW'
    }

    county2idx = {
        'CON': 6,
        'DEV': 2,
        'DUR': 4,
        'ELN': 15,
        'KEN': 9,
        'LAN': 7,
        'LEI': 17,
        'LND': 0,
        'MDX': 13,
        'MLN': 11,
        'NBL': 8,
        'NTT': 3,
        'OXF': 12,
        'SOM': 5,
        'WES': 16,
        'WIL': 10,
        'WLN': 14,
        'YKS': 1
    }

    idx2county = {
        0: 'LND',
        1: 'YKS',
        2: 'DEV',
        3: 'NTT',
        4: 'DUR',
        5: 'SOM',
        6: 'CON',
        7: 'LAN',
        8: 'NBL',
        9: 'KEN',
        10: 'WIL',
        11: 'MLN',
        12: 'OXF',
        13: 'MDX',
        14: 'WLN',
        15: 'ELN',
        16: 'WES',
        17: 'LEI'
    }


if __name__ == '__main__':
    datadir = 'C:/Users/ryanc/Documents/corpora/fred_s'

    m = get_files_labels(datadir)

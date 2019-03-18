import config
import pandas as pd

from common import get_logger
from enum import Enum

logger = get_logger('RL.Base.LoadCensus')

class CensusLocation(Enum):
    """Enumerator for different census towns/villages"""
    SANT_FELIU = 1
    CATELLVI_ROSANES = 2
    SANTA_COLOMA = 3

#Map of census towns to their XLSX files.
census_location_to_file_map = {
    CensusLocation.SANT_FELIU : config.CENSUS_SANT_FELIU,
    CensusLocation.CATELLVI_ROSANES : config.CENSUS_CATELLVI,
    CensusLocation.SANTA_COLOMA : config.CENSUS_SANTA_COLOMA
}

class Harmonization(object):
    """Class to harmonize certain fields."""
    male_name_map = {}
    female_name_map = {}
    surname_map = {}
    relationship_map = {}
    occupation_map = {}

    def __init__(self, male_name_file='data/harmo/male_name.tsv',
                    female_name_file='data/harmo/female_name.tsv',
                    surname_file='data/harmo/surname.tsv',
                    relationship_file='data/harmo/relation.tsv',
                    occupation_file='data/harmo/occupation.tsv'):
        with open(male_name_file, "r") as f:
            for line in f.readlines():
                parts = line.split('\t')
                self.male_name_map[parts[0]] = parts[1].strip()

        with open(female_name_file, "r") as f:
            for line in f.readlines():
                parts = line.split('\t')
                self.female_name_map[parts[0]] = parts[1].strip()

        with open(surname_file, "r") as f:
            for line in f.readlines():
                parts = line.split('\t')
                self.surname_map[parts[0]] = parts[1].strip()

        with open(relationship_file, "r") as f:
            for line in f.readlines():
                parts = line.split('\t')
                self.relationship_map[parts[0]] = parts[1].strip()

        with open(occupation_file, "r") as f:
            for line in f.readlines():
                parts = line.split('\t')
                self.occupation_map[parts[0]] = parts[1].strip()


def load_census(census_location=CensusLocation.SANT_FELIU, keep_default_na=False,
                years=[1940, 1936, 1930, 1924, 1920, 1915],
                filters=[(lambda x: x['any_padro'] is not None)],
                fields=['id_padro_individu','any_padro', 'DNI', 'Noms_harmo', 'cognom1_harmo', 'bg',
                        'cognom2_harmo', 'cohort', 'estat_civil', 'parentesc_har', 'ocupacio_hisco']):
    """
        Base Method to read data from XLSX census files into pandas Dataframe
        :param census_location: Enum value to locate XLSX file.
        :param keep_default_na: Boolean flag to read empty cells as NA
        :param years: List of years (in integers) to restrict input.
        :param filters: List of lambda functions to apply to dataFrame
        :param fields: List of column names to return. Set None to get all fields.
    """
    assert census_location in CensusLocation, "census_location is not defined in Enum"
    filename = census_location_to_file_map[census_location]
    WS = pd.read_excel(filename, keep_default_na=keep_default_na)
    filtered_data = pd.DataFrame()
    for y in years:
        filtered_data = filtered_data.append(WS[WS.any_padro == y])
    for f in filters:
        filtered_data = filtered_data[filtered_data.apply(f, axis=1)]
    if fields:
        for f in fields:
            if f not in filtered_data.keys():
                logger.info("Incorrect Field requested: %s. Removing it.", str(f))
                fields.remove(f)
        return filtered_data[fields]
    return filtered_data
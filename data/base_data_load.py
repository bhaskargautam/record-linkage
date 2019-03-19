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

#Map of census towns to their sheet name in XLSX files.
census_location_to_sheet_name_map = {
    CensusLocation.SANT_FELIU : 0,
    CensusLocation.CATELLVI_ROSANES : 1,
    CensusLocation.SANTA_COLOMA : 1
}

#Map of census towns to the column name for census year
census_location_to_year_column_map = {
    CensusLocation.SANT_FELIU : 'any_padro',
    CensusLocation.CATELLVI_ROSANES : 'Padro_any',
    CensusLocation.SANTA_COLOMA : 'Padro_any'
}

census_location_to_gender_column_map = {
    CensusLocation.SANT_FELIU : None,
    CensusLocation.CATELLVI_ROSANES : 'Persona_sexe',
    CensusLocation.SANTA_COLOMA : 'Persona_sexe'
}

class HarmonizeFields(Enum):
    """Enumerator for different fields which can be harmonized"""
    MALE_NAME = 1
    FEMALE_NAME = 2
    SURNAME = 3
    RELATION = 4
    OCCUPATION = 5

class Harmonization(object):
    """Class to harmonize certain fields."""

    def __init__(self, male_name_file='data/harmo/male_name.tsv',
                    female_name_file='data/harmo/female_name.tsv',
                    surname_file='data/harmo/surname.tsv',
                    relationship_file='data/harmo/relation.tsv',
                    occupation_file='data/harmo/occupation.tsv'):
        self.male_name_map = {}
        self.female_name_map = {}
        self.surname_map = {}
        self.relationship_map = {}
        self.occupation_map = {}

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

        self.map_for_fields = {
            HarmonizeFields.MALE_NAME: self.male_name_map,
            HarmonizeFields.FEMALE_NAME: self.female_name_map,
            HarmonizeFields.SURNAME: self.surname_map,
            HarmonizeFields.RELATION: self.relationship_map,
            HarmonizeFields.OCCUPATION: self.occupation_map
        }

    def get_harmonization(self, harmonize_field, value):
        try:
            return self.map_for_fields[harmonize_field][value]
        except KeyError:
            return value

#Create a static object of Harmonization Class
harmonization = Harmonization()

def load_census(census_location=CensusLocation.SANT_FELIU, keep_default_na=False,
                years=[1940, 1936, 1930, 1924, 1920, 1915],
                filters=[(lambda x: x['any_padro'] is not None)],
                fields=['id_padro_individu','any_padro', 'DNI', 'Noms_harmo', 'cognom1_harmo', 'bg',
                        'cognom2_harmo', 'cohort', 'estat_civil', 'parentesc_har', 'ocupacio_hisco'],
                harmonize_fields={HarmonizeFields.SURNAME : 'cognom_1'}):
    """
        Base Method to read data from XLSX census files into pandas Dataframe
        :param census_location: Enum value to locate XLSX file.
        :param keep_default_na: Boolean flag to read empty cells as NA
        :param years: List of years (in integers) to restrict input. Set None to get all.
        :param filters: List of lambda functions to apply to dataFrame
        :param fields: List of column names to return. Set None to get all fields.
        :param harmonize_fields: Dictionary containing fields for harmonization.
        :return DataFrame containing census data after applying filters.
        :rtype: pandas.DataFrame
    """
    assert census_location in CensusLocation, "census_location is not defined in Enum"

    #Read data from XLSX file
    filename = census_location_to_file_map[census_location]
    sheet_name = census_location_to_sheet_name_map[census_location]
    WS = pd.read_excel(filename, keep_default_na=keep_default_na, sheet_name=sheet_name)

    year_field_name = census_location_to_year_column_map[census_location]
    #Filter data by years.
    filtered_data = pd.DataFrame()
    if years:
        for y in years:
            filtered_data = filtered_data.append(WS[WS[year_field_name] == y])

    #Apply custom filters
    if filters:
        for f in filters:
            filtered_data = filtered_data[filtered_data.apply(f, axis=1)]

    #Harmonize requested fields
    new_fields = []
    if harmonize_fields:
        for key in harmonize_fields:
            base_field_name = harmonize_fields[key]
            new_field_name = harmonize_fields[key] + '_harmo'
            new_fields.append(new_field_name)

            if key in [HarmonizeFields.MALE_NAME, HarmonizeFields.FEMALE_NAME]:
                gender_column = census_location_to_gender_column_map[census_location]
                assert gender_column is not None, "Cannot Harmonize Name, since gender not known"

                filtered_data[new_field_name] = filtered_data.apply(lambda x: \
                    harmonization.get_harmonization(HarmonizeFields.MALE_NAME \
                        if x[gender_column] == 'H' else HarmonizeFields.FEMALE_NAME,
                        x[base_field_name]), axis=1)
            else:
                filtered_data[new_field_name] = filtered_data.apply(lambda x: \
                    harmonization.get_harmonization(key, x[base_field_name]), axis=1)

    #Select requested fields
    if fields:
        fields.extend(harmonize_fields.values())
        fields.extend(new_fields)
        fields = list(set(fields))
        for f in fields:
            if f not in filtered_data.keys():
                logger.info("Incorrect Field requested: %s. Removing it.", str(f))
                fields.remove(f)
        filtered_data = filtered_data[fields]

    return filtered_data
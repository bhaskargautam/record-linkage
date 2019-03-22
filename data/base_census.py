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


class HarmonizeFields(Enum):
    """Enumerator for different fields which can be harmonized"""
    MALE_NAME = 1
    FEMALE_NAME = 2
    SURNAME_1 = 3
    SURNAME_2 = 4
    RELATION = 5
    OCCUPATION = 6

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
            HarmonizeFields.SURNAME_1: self.surname_map,
            HarmonizeFields.SURNAME_2: self.surname_map,
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

#Map of census towns to the columns for harmonization
census_location_to_harmonize_fields_map = {
    CensusLocation.SANT_FELIU : None,
    CensusLocation.CATELLVI_ROSANES: {
        HarmonizeFields.SURNAME_1 : 'Persona_cognom1',
        HarmonizeFields.SURNAME_2 : 'Persona_cognom2',
        HarmonizeFields.MALE_NAME : 'Persona_nom',
        HarmonizeFields.RELATION : 'Persona_parentesc',
        HarmonizeFields.OCCUPATION : 'Persona_ofici'
    },
    CensusLocation.SANTA_COLOMA: {
        HarmonizeFields.SURNAME_1 : 'Persona_cognom1',
        HarmonizeFields.SURNAME_2 : 'Persona_cognom2',
        HarmonizeFields.MALE_NAME : 'Persona_nom',
        HarmonizeFields.RELATION : 'Persona_parentesc',
        HarmonizeFields.OCCUPATION : 'Persona_ofici'
    }
}


class CensusFields(Enum):
    FIRST_NAME = 1
    SURNAME_1 = 2
    SURNAME_2 = 3
    ID_INDIVIDUAL = 4
    ID_HOUSEHOLD = 5
    CENSUS_YEAR = 6
    DNI = 7
    YOB = 8
    CIVIL_STATUS = 9
    RELATION = 10
    OCCUPATION = 11
    GENDER = 12
    DOB = 13
    AGE = 14

census_field_map = {
    CensusLocation.SANT_FELIU: {
        CensusFields.FIRST_NAME : 'Noms_harmo',
        CensusFields.SURNAME_1 : 'cognom_1',
        CensusFields.SURNAME_2 : 'cognom_2',
        CensusFields.ID_INDIVIDUAL : 'id_padro_individu',
        CensusFields.ID_HOUSEHOLD: 'id_padro_llar',
        CensusFields.CENSUS_YEAR : 'any_padro',
        CensusFields.DNI : 'DNI',
        CensusFields.YOB : 'cohort',
        CensusFields.CIVIL_STATUS : 'estat_civil',
        CensusFields.RELATION: 'parentesc_har',
        CensusFields.OCCUPATION : 'ocupacio_hisco',
        CensusFields.GENDER : None,
        CensusFields.DOB : 'data_naix',
        CensusFields.AGE : 'edat'
    },
    CensusLocation.SANTA_COLOMA : {
        CensusFields.FIRST_NAME : 'Persona_nom_harmo',
        CensusFields.SURNAME_1 : 'Persona_cognom1_harmo',
        CensusFields.SURNAME_2 : 'Persona_cognom2_harmo',
        CensusFields.ID_INDIVIDUAL : None,
        CensusFields.ID_HOUSEHOLD: None,
        CensusFields.CENSUS_YEAR : 'Padro_any',
        CensusFields.DNI : 'DNI',
        CensusFields.YOB : None,
        CensusFields.CIVIL_STATUS : 'Persona_estatcivil',
        CensusFields.RELATION: 'Persona_parentesc_harmo',
        CensusFields.OCCUPATION : 'Persona_ofici_harmo',
        CensusFields.GENDER : 'Persona_sexe',
        CensusFields.DOB: 'Persona_data_naix',
        CensusFields.AGE : 'Persona_edata',
    },
    CensusLocation.CATELLVI_ROSANES : {
        CensusFields.FIRST_NAME : 'Persona_nom_harmo',
        CensusFields.SURNAME_1 : 'Persona_cognom1_harmo',
        CensusFields.SURNAME_2 : 'Persona_cognom2_harmo',
        CensusFields.ID_INDIVIDUAL : None,
        CensusFields.ID_HOUSEHOLD: None,
        CensusFields.CENSUS_YEAR : 'Padro_any',
        CensusFields.DNI : 'DNI',
        CensusFields.YOB : None,
        CensusFields.CIVIL_STATUS : 'Persona_estatcivil',
        CensusFields.RELATION: 'Persona_parentesc_harmo',
        CensusFields.OCCUPATION : 'Persona_ofici_harmo',
        CensusFields.GENDER : 'Persona_sexe',
        CensusFields.DOB: 'Persona_data_naix',
        CensusFields.AGE : 'Persona_edata',
    }
}

def load_census(census_location=CensusLocation.SANT_FELIU, keep_default_na=False,
                years=[1940, 1936, 1930, 1924, 1920, 1915],
                filters=[(lambda x: str.isdigit(str(x['DNI'])) or x['DNI'] == '')],
                fields=[CensusFields.ID_INDIVIDUAL,CensusFields.CENSUS_YEAR, CensusFields.DNI,
                        CensusFields.FIRST_NAME, CensusFields.SURNAME_1, CensusFields.SURNAME_2,
                        CensusFields.YOB, CensusFields.CIVIL_STATUS, CensusFields.RELATION,
                        CensusFields.OCCUPATION, CensusFields.GENDER]):
    """
        Base Method to read data from XLSX census files into pandas Dataframe
        :param census_location: Enum value to locate XLSX file.
        :param keep_default_na: Boolean flag to read empty cells as NA
        :param years: List of years (in integers) to restrict input. Set None to get all.
        :param filters: List of lambda functions to apply to dataFrame
        :param fields: List of column names to return. Set None to get all fields.
        :return DataFrame containing census data after applying filters.
        :rtype: pandas.DataFrame
    """
    assert census_location in CensusLocation, "census_location is not defined in Enum"

    #Read data from XLSX file
    filename = census_location_to_file_map[census_location]
    sheet_name = census_location_to_sheet_name_map[census_location]
    WS = pd.read_excel(filename, keep_default_na=keep_default_na, sheet_name=sheet_name)
    logger.info("Shape of Excel Worksheet: %s", str(WS.shape))

    year_field_name = census_field_map[census_location][CensusFields.CENSUS_YEAR]
    #Filter data by years.
    if years:
        filtered_data = WS[WS[year_field_name].isin(years)]
    else:
        filtered_data = WS
    logger.info("Shape of Filtered Data by years: %s", str(filtered_data.shape))

    #Create DNI column if missing in data.
    if 'DNI' not in filtered_data.keys():
        filtered_data['DNI'] = pd.Series([None]* filtered_data.shape[0])

    #Apply custom filters
    if filters:
        for f in filters:
            filtered_data = filtered_data[filtered_data.apply(f, axis=1)]
    logger.info("Shape of Filtered Data after custom func: %s", str(filtered_data.shape))

    #Harmonize requested fields
    new_fields = []
    harmonize_fields = census_location_to_harmonize_fields_map[census_location]
    if harmonize_fields:
        for key in harmonize_fields:
            base_field_name = harmonize_fields[key]
            new_field_name = harmonize_fields[key] + '_harmo'
            new_fields.append(new_field_name)

            if key in [HarmonizeFields.MALE_NAME, HarmonizeFields.FEMALE_NAME]:
                gender_column = census_field_map[census_location][CensusFields.GENDER]
                assert gender_column is not None, "Cannot Harmonize Name, since gender not known"

                filtered_data[new_field_name] = filtered_data.apply(lambda x: \
                        harmonization.get_harmonization(HarmonizeFields.MALE_NAME, x[base_field_name])\
                        if x[gender_column] == 'H' else harmonization.get_harmonization(\
                            HarmonizeFields.FEMALE_NAME, x[base_field_name]),axis=1)
            else:
                filtered_data[new_field_name] = filtered_data.apply(lambda x: \
                    harmonization.get_harmonization(key, x[base_field_name]), axis=1)

    logger.info("Shape of Filtered Data after harmonization: %s", str(filtered_data.shape))

    #Select requested fields
    if fields:
        fields = [census_field_map[census_location][f] or f for f in fields]
        fields.extend(new_fields)

        #if harmonize_fields:
        #    fields.extend(harmonize_fields.values())

        fields = list(set(fields))
        for f in fields:
            if f not in filtered_data.keys():
                logger.info("Incorrect Field requested: %s. Removing it.", str(f))
                fields.remove(f)
        filtered_data = filtered_data[fields]

    logger.info("Shape of Filtered Data after selecting fields: %s", str(filtered_data.shape))
    return filtered_data
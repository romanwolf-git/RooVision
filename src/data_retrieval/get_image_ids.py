import json
import os
import random

import galah


def get_species_names(taxa='Macropodidae', state='Victoria'):
    """
    Get a list of species in a specified Australian state and taxa from the Atlas of Living Australia

    :param str state: Australian State of interest (default: 'Victoria')
    :param str taxa: formal scientific name of grouped species (default: 'Macropodidae')
    :return: accepted species of the given taxa in an Australian state of interest
    :rtype: list

    . warning::

        This function only runs if you have a registered account at the Atlas of Living Australia.
        To register go to https://www.ala.org.au/

    """
    state_filter = 'stateProvince=' + state
    df_species = galah.atlas_species(taxa=taxa,
                                     filters=state_filter)
    if df_species.empty:
        species_names = None
        print(f'there are no records of {taxa} in {state}.')
    else:
        species_names = list(df_species['Species Name'])

    return species_names


def get_image_ids(species_names, **kwargs):
    """
    Get the image IDs for one or more species from the Atlas of Living Australia in an Australian state.

    :param str or list species_names: Scientific name(s) of species, cannot be 'None'.
    :param str state: Australian state of interest (default: 'Victoria').
    :return: A dictionary containing species names as keys and corresponding lists of image IDs as values.
             Returns None if no image IDs are found.
    :rtype: dict or None
    """
    fields = ['scientificName', 'dataResourceName', 'occurrenceStatus', 'multimedia', 'images', 'videos']

    # check if there is at least one species
    if isinstance(species_names, type(None)):
        print('Provide at least one species or a list of species, not "None".')
        return {}

    if not isinstance(species_names, list):
        # Convert single species to a list
        species_names = [species_names]

    # create dictionary with species: list(image_ids)
    image_id_dict = dict()
    for idx, species in enumerate(species_names):
        # create a dataframe for species with fields and state
        df_images = galah.atlas_occurrences(taxa=species, fields=fields)

        if df_images.empty:
            print(f'There are no media for {species} in the Atlas of Living Australia.')

            if idx + 1 < len(species_names):
                continue
            else:
                return {}

        # get image ids
        image_ids = df_images['images'].dropna()
        if len(image_ids) > 0:
            image_ids = image_ids.str.split('|', expand=True).values.flatten()
            image_ids = [image_id.strip() for image_id in image_ids if image_id is not None]
            image_id_dict[species] = image_ids
        else:
            print(f'There are no images for {species} in the Atlas of Living Australia.')

    if not image_id_dict:
        return {}
    else:
        for species, image_ids in image_id_dict.items():
            print(f'There are {len(image_ids)} images of {species} in the database.')
        return image_id_dict


def get_randomized_image_urls(image_id_dict, n_sample=200):
    """
    this function randomly samples a given number of ids, composes urls and returns a dictionary of species and
    image_ids

    :param int n_sample: number of samples to pick (default: 200)
    :param dict image_id_dict: a dictionary with key: species, value: list of image_ids
    :return: dictionary with species and image-urls
    """
    random.seed(42)
    base_url = 'https://images.ala.org.au/ws/image/{id}'

    image_url_dict = dict()

    if image_id_dict:
        for species, image_ids in image_id_dict.items():

            # randomly sample if there are more than n_sample ids available
            if len(image_ids) >= n_sample:
                image_ids = random.sample(image_ids, n_sample)
                urls = [base_url.replace('{id}', media_id) for media_id in image_ids]

            else:
                urls = [base_url.replace('{id}', media_id) for media_id in image_ids]

            image_url_dict[species] = urls

        return image_url_dict


def sort_dict(image_dict):
    """
    sorts the dictionary according to the number of available images
    :param dict image_dict: a dictionary with key: species, value: list of image_ids
    :return: sorted dictionary
    """
    return dict(sorted(image_dict.items(), reverse=True, key=lambda item: len(item[1])))


def save_dictionary(image_dict, file_name='image_urls.json'):
    """
    save the dictionary as json-file
    :param dict image_dict: a dictionary with key: species, value: list of image_ids
    :param str file_name: name of the json-file, i.e. must contain ".json" (default: "image_urls.json")
    """
    complete_file_name = os.path.join(os.getcwd(), file_name)

    with open(complete_file_name, 'w') as json_file:
        json.dump(image_dict, json_file)
        print(f'saved {file_name} to {os.getcwd()}')


if __name__ == "__main__":
    species_names = get_species_names()
    image_id_dict = get_image_ids(species_names)
    image_url_dict = sort_dict(get_randomized_image_urls(image_id_dict))
    save_dictionary(image_url_dict)

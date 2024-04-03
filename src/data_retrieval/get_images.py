import asyncio
import json
import logging
import pickle
import inspect
from time import perf_counter
from urllib.parse import urlparse

import aiofiles
import aiohttp

# Set up logging
logging.basicConfig(filename='data/logs/download_log.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set headers for the GET requests
HEADERS = {'User-Agent': ''}


async def fetch_original_image_url(session, url):
    """
    Fetch the original image URL from a given URL on the Atlas of Living Australia.

    :param aiohttp.ClientSession session: Aiohttp client session.
    :param str url: The URL to fetch data from.

    :return: str: Original filename retrieved from the URL.
    """
    if check_url(url):
        async with session.get(url, headers=HEADERS) as response:
            try:
                response.raise_for_status()
                data = await response.json()
                return data['originalFileName']
            except (aiohttp.ClientError, json.JSONDecodeError) as e:
                logger.error(f'Error fetching data: {e}')


async def get_tasks(coroutine, session, urls, **kwargs):
    """
    Schedule given coroutines to run concurrently for a list of URLs.

    :param coroutine: Coroutines to be scheduled.
    :param aiohttp.ClientSession session: Aiohttp client session.
    :param iterable urls: Iterable of image URLs.

    :return: An awaitable object that represents the combined results of the provided coroutines.
             This object is typically an instance of the asyncio.Future class.
    """
    tasks = list()

    for idx, url in enumerate(urls):
        if 'i' in inspect.signature(coroutine).parameters:
            # coroutine supports the index 'i' argument
            task = coroutine(session, url, **kwargs, i=idx)
        else:
            # coroutine without index
            task = coroutine(session, url, **kwargs)

        tasks.append(task)

    return await asyncio.gather(*tasks)


def read_pickle_dict(file_name):
    """
    Read pickled image-URL dictionary.

    :param str file_name: Name of the file to read (default: 'species_image_ids.pkl').
    :return: dict: Loaded dictionary from the pickled file.
    """
    with open(file_name, 'rb') as file:
        image_url_dict = pickle.load(file)
    return image_url_dict


def read_json_image_urls(file_name):
    """
    Reads image-URLs from JSON.

    :param str file_name: Name of the file to read (default: 'species_image_ids.json').
    :return: dict: Loaded dictionary from the JSON file.
    """
    with open(file_name, 'r') as file:
        image_url_dict = json.load(file)
        image_url_dict = {key: image_url_dict[key] for idx, key in enumerate(image_url_dict) if idx < 100}
    return image_url_dict


async def save_image(session, url, species, i):
    """
    Save images asynchronously.

    :param str url: Original image URL.
    :param aiohttp.ClientSession session: Aiohttp client session.
    :param str species: Name of Species.
    :param int i: Index for file labelling.

    :return: None
    """
    if check_url(url):
        async with session.get(url, headers=HEADERS) as resp:
            try:
                resp.raise_for_status()
                if resp.status == 200:
                    file_name = f"data/images/raw/{species.lower().replace(' ', '_')}_{i:03d}.jpg"

                    async with aiofiles.open(file_name, mode='wb') as f:
                        await f.write(await resp.read())
                        logger.info(f'Saved {file_name.split("/")[-1]} of {species}.')
                else:
                    logger.error(f'Error downloading {url} for {species}, status code: {resp.status}')
            except aiohttp.ClientError as e:
                logger.error(f'Aiohttp client error: {e}')
            except Exception as e:
                logger.error(f'Error during image download: {e}')


def check_url(url):
    """
    Checks if the given URL is valid and if it is from a valid host.

    :param str url: File name to check.
    :return bool: True if the URL is a valid, False otherwise.
    """
    # Check for specific invalid hostnames
    invalid_hosts = ['media.bowerbird.org.au', 'fielddata.ala.org.au']

    if url is not None:
        # Check if the file_name is a valid URL
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            # Check if the hostname is in the list of invalid hosts
            if parsed_url.netloc in invalid_hosts:
                logger.warning(f'{parsed_url.netloc} is an invalid host.')
                return False
            else:
                return True

    # Log a warning for invalid URLs
    logger.warning(f'Invalid URL: {url}')
    return False


async def download_images(image_url_dict):
    """
    Download images asynchronously based on species and image IDs.
    """
    for species, image_urls in image_url_dict.items():
        async with aiohttp.ClientSession(trust_env=True) as session:
            original_image_urls = await get_tasks(coroutine=fetch_original_image_url,
                                                  session=session,
                                                  urls=image_urls)

            await get_tasks(coroutine=save_image,
                            session=session,
                            urls=original_image_urls,
                            species=species  # kwarg
                            )


if __name__ == '__main__':
    # Start measuring script execution time
    start = perf_counter()

    # Create a new asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Read image URLs from JSON file
    image_url_dict = read_json_image_urls('data/json/roo_image_urls.json')

    # Run asynchronous image download tasks
    loop.run_until_complete(download_images(image_url_dict))

    # Stop measuring script execution time
    stop = perf_counter()

    # Calculate elapsed time
    elapsed_time = stop - start
    elapsed_time_minutes = elapsed_time / 60.0

    # Log elapsed time
    logger.info(f"Time taken: {elapsed_time_minutes:.2f} minutes")

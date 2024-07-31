import json


##### Funciones para leer archivos json #####

def read_json(file,encoding="utf8"):
    """ Function to read json files

    Args:
        file (str): path to json file
        encoding (str): encoding of the file

    Returns:
        dict: json data
    """
    with open(file,encoding=encoding) as f: # VER LOS ENCODINGS DE LOS ARCHIVOS  Latin-1, UTF-8
        data = json.load(f)
    return data
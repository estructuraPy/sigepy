
def get_file_location(file_name: str):
    notebook_location = pathlib.Path().absolute()
    parent_directory = notebook_location.parent
    data_folder = parent_directory / 'tests/test_data'
    file_location = data_folder / file_name
    return file_location
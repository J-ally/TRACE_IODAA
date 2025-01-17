#!/usr/bin/python
# -*- coding: utf-8 -*-

# creator : Joseph Allyndree
# date : 2024-01-17

import os
from sys import platform
from copy import copy


def test_import () -> bool:
    """
    Returns True if the file is imported correctly.
    Returns:
        True (bool): True if the file is imported correctly.
    """
    print("UTILS_files.py imported successfully")
    return True

###############################################################################
#                             FILES MANAGEMENT                                #
###############################################################################


def filter_out_system_files (files_list : list[str]) -> list[str] :
    """
    Check if the files are system files and remove them from the list
    The following files will also be excluded :
        - files beginning with "."
        - files beginning with "$"
        - files beginning with "._"
        - files named "System Volume Information"
    Args:
        files_list (list[str]): the list of files to filter

    Returns:
        list of str: the resulting list of strings
    """
    indexes_to_pop = []
    for file in files_list :
        if file.split(f"{os.sep}")[-1].startswith(".") : 
            indexes_to_pop.append(files_list.index(file))
        elif file.split(f"{os.sep}")[-1].startswith("._") :
            indexes_to_pop.append(files_list.index(file))
        elif file.split(f"{os.sep}")[-1].startswith("$") :
            indexes_to_pop.append(files_list.index(file))
        elif "System Volume Information" in file.split(f"{os.sep}") :
            indexes_to_pop.append(files_list.index(file))
    
    for index in sorted(indexes_to_pop, reverse=True): # remove the files in reverse order to not mess up the indexes
        del files_list[index]

    return files_list


def get_accelero_id_from_parquet_or_csv_files (path_parquet_or_csv: str, only_id : str = False) -> str :
    """
    Retrieve the id of the accelerometer from the path.

    Args:
        path_parquet_or_csv (str): the path to the accelerometer data
        only_id (bool, optional): if True, only the id is returned. Defaults to False.
    Returns:
        str: the id of the accelerometer
    """
    
    if path_parquet_or_csv.endswith(".parquet") or path_parquet_or_csv.endswith(".csv") :
        nom_parquet = path_parquet_or_csv.split(os.sep)[-1]
        id_accelero_long = nom_parquet.split("_")[1]
        if only_id :
            id_accelero = id_accelero_long[-4:]
            return id_accelero
        else :
            return id_accelero_long
    else :
        raise ValueError(f"The file {path_parquet_or_csv} is not a parquet nor a csv")
    

def get_all_files_within_one_folder (folder_to_search : str, ordered_by_id : bool, extension : str = "all") -> list[str] :
    """
    This function is used to retrieve all the files from the directories.
    
    Args:
        folder_to_search (str) : the folder to search for the files
        ordered_by_id (bool) :  if True, the files will be ordered by the full id of the accelerometer (e.g. d0003cec)
                                if False, the files will be ordered by the name of the file
        extension (str) defaults to "all" : the extension of the files to search for
            can be any type of extension file (e.g. ".dat", or ".csv") or "all" meaning no filter is made on the files being returned
    Retuns:
        list[str] : the list of files paths ordered or not
    """
    files_ = [os.path.join(folder_to_search,f) for f in os.listdir(folder_to_search) if os.path.isfile(os.path.join(folder_to_search,f))]
    files_filtered = filter_out_system_files(files_)
    
    if extension != "all" :
        files_filtered = [file for file in files_filtered if file.endswith(extension)]
    
    if ordered_by_id : 
        if files_filtered[0].endswith(".parquet") or files_filtered[0].endswith(".csv") :
            dict_files = {get_accelero_id_from_parquet_or_csv_files(file) : file for file in files_filtered}
        files_filtered = [dict_files[key] for key in sorted(dict_files.keys())]
    
    return files_filtered
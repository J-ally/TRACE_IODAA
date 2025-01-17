# -*- coding: utf-8 -*-
# creator : Joseph Allyndree
# date : 2025-01-17

###############################################################################
#                                .ENV TEMPLATE                                #
###############################################################################

# parent_dir="" # path to the folder where the data is stored
# output_dir = "" # path the folder where the output will be stored

###############################################################################
#                           GET GLOBAL VARIABLES                              #
###############################################################################

import os
from dotenv import load_dotenv

###############################################################################
#                           STANDARDIZE PATHING                               #
###############################################################################

load_dotenv()
parent_dir_raw = os.getenv("parent_folder")
parent_dir = os.sep.join([parent_dir_raw])

output_dir_raw = os.getenv("output_dir")
output_dir = os.sep.join([output_dir_raw])
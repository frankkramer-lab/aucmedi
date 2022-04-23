#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                    Documentation                    #
#-----------------------------------------------------#
""" The IO Interfaces of AUCMEDI allow extracting information from datasets in different
    structures or formats.

These interfaces are called **internally** via the [input_interface][aucmedi.data_processing.io_data.input_interface].

???+ info "Format Interfaces"
    | Interface      | Internal Function                                                    | Description                                  |
    | -------------- | -------------------------------------------------------------------- | -------------------------------------------- |
    |  `"csv"`       | [io_csv()][aucmedi.data_processing.io_interfaces.io_csv]             | Storing class annotations in a CSV file.     |
    |  `"directory"` | [io_directory()][aucmedi.data_processing.io_interfaces.io_directory] | Storing class annotations in subdirectories. |
    |  `"json"`      | [io_json()][aucmedi.data_processing.io_interfaces.io_json]           | Storing class annotations in a JSON file.    |

"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.data_processing.io_interfaces.io_csv import csv_loader
from aucmedi.data_processing.io_interfaces.io_json import json_loader
from aucmedi.data_processing.io_interfaces.io_directory import directory_loader

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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
# Internal libraries
#-----------------------------------------------------#
#                     Parser - CLI                    #
#-----------------------------------------------------#
def parse_cli(args):
    """ Internal function for parsing CLI arguments to a valid configuration
    dictionary.
    """
    # Parse argument namespace to config dict
    config = vars(args)

    # Convert variables
    if config["hub"] == "training":
        # Handle 3D shape - from str to tuple
        config["shape_3D"] = tuple(map(int, config["shape_3D"].split("x")))
    else:
        # Handle input path
        config["input"] = config["path_input"]
        del config["path_input"]
    # Handle output path
    config["output"] = config["path_output"]
    del config["path_output"]

    # Return valid configs
    return config

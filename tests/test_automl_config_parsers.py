#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
# Python Standard Library
import json
from os import path
# Third Party Libraries
import unittest
import tempfile
import yaml
from argparse import Namespace
import copy
from pydantic import ValidationError
import yaml.parser
# Internal Libraries
from aucmedi.automl.config_parsers import parse_config_file
#-----------------------------------------------------#
#           Unittest: AutoML Config Parsers           #
#-----------------------------------------------------#
class AutoML_ConfigParser(unittest.TestCase):
    # Create configs to be checked against
    @classmethod
    def setUpClass(self):
        # load training example config and setup configs to be saved
        with open('./examples/example_configs/config.yaml', 'r') as file:
            training_config = yaml.safe_load(file)
        # setup prediction example config
        prediction_config = copy.deepcopy(training_config)
        prediction_config['general']['hub'] = 'prediction'
        # setup evaluation example config
        evaluation_config = copy.deepcopy(training_config)
        evaluation_config['general']['hub'] = 'evaluation'
        # build wrong configs
        wrong_base_config = _make_config_erroneous(training_config, 'general')
        wrong_training_config = _make_config_erroneous(training_config, 'training')
        wrong_prediction_config = _make_config_erroneous(prediction_config, 'prediction')
        wrong_evaluation_config = _make_config_erroneous(evaluation_config, 'evaluation')
        # setup temporary directory for config files during tests
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix='tmp.aucmedi.', suffix='.config'
        )
        # save correct yml configs
        _save_config_to_file(training_config, 'yml', path.join(self.tmp_data.name, 'training_config.yml'))
        _save_config_to_file(prediction_config, 'yml', path.join(self.tmp_data.name, 'prediction_config.yml'))
        _save_config_to_file(evaluation_config, 'yml', path.join(self.tmp_data.name, 'evaluation_config.yml'))
        # save correct json configs
        _save_config_to_file(training_config, 'json', path.join(self.tmp_data.name, 'training_config.json'))
        _save_config_to_file(prediction_config, 'json', path.join(self.tmp_data.name, 'prediction_config.json'))
        _save_config_to_file(evaluation_config, 'json', path.join(self.tmp_data.name, 'evaluation_config.json'))
        # save incorrect base yaml and json config
        _save_config_to_file(wrong_base_config, 'yml', path.join(self.tmp_data.name, 'wrong_base_config.yml'))
        _save_config_to_file(wrong_base_config, 'json', path.join(self.tmp_data.name, 'wrong_base_config.json'))
        # save incorrect yaml configs
        _save_config_to_file(wrong_training_config, 'yml', path.join(self.tmp_data.name, 'wrong_training_config.yml'))
        _save_config_to_file(wrong_prediction_config, 'yml', path.join(self.tmp_data.name, 'wrong_prediction_config.yml'))
        _save_config_to_file(wrong_evaluation_config, 'yml', path.join(self.tmp_data.name, 'wrong_evaluation_config.yml'))
        # save incorrect json configs
        _save_config_to_file(wrong_training_config, 'json', path.join(self.tmp_data.name, 'wrong_training_config.json'))
        _save_config_to_file(wrong_prediction_config, 'json', path.join(self.tmp_data.name, 'wrong_prediction_config.json'))
        _save_config_to_file(wrong_evaluation_config, 'json', path.join(self.tmp_data.name, 'wrong_evaluation_config.json'))
        # setup self.configs for usage during validation
        self.training_config = training_config['training']
        self.training_config['shape_3D'] = tuple(self.training_config['shape_3D'])
        self.training_config['hub'] = 'training'
        self.training_config['path_imagedir'] = training_config['general']['path_imagedir']
        self.prediction_config = prediction_config['prediction']
        self.prediction_config['hub'] = 'prediction'
        self.prediction_config['path_imagedir'] = prediction_config['general']['path_imagedir']
        self.evaluation_config = evaluation_config['evaluation']
        self.evaluation_config['hub'] = 'evaluation'
        self.evaluation_config['path_imagedir'] = evaluation_config['general']['path_imagedir']
    #-------------------------------------------------#
    #             parse config file: YAML             #
    #-------------------------------------------------#
    def test_parse_yaml_training_config_with_correct_config(self):
        # test if yaml training config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'training_config.yml')
        ) 
        parsed_config = parse_config_file(args, 'yml')
        for key, value in self.training_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_yaml_prediction_config_with_correct_config(self):
        # test if yaml prediction config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'prediction_config.yml')
        ) 
        parsed_config = parse_config_file(args, 'yml')
        for key, value in self.prediction_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_yaml_evaluation_config_with_correct_config(self):
        # test if yaml evaluation config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'evaluation_config.yml')
        ) 
        parsed_config = parse_config_file(args, 'yml')
        for key, value in self.evaluation_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_yaml_base_config_with_wrong_config(self):
        # test if yaml base config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_base_config.yml')
        )
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'yml')
        # completely wrong base config should have 2/2 validation errors
        self.assertEqual(2, len(ve.exception.errors()))
    def test_parse_yaml_training_config_with_wrong_config(self):
        # test if yaml training config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_training_config.yml')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'yml')
        # completely wrong training config should have 11/11 validation errors
        self.assertEqual(11, len(ve.exception.errors()))
    def test_parse_yaml_prediction_config_with_wrong_config(self):
        # test if yaml prediction config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_prediction_config.yml')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'yml')
        # completely wrong prediction config should have 6/6 validation errors
        self.assertEqual(6, len(ve.exception.errors()))
    def test_parse_yaml_evaluation_config_with_wrong_config(self):
        # test if yaml evaluation config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_evaluation_config.yml')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'yml')
        # completely wrong evaluation config should have 4/4 validation errors
        self.assertEqual(4, len(ve.exception.errors()))
    #-------------------------------------------------#
    #             parse config file: JSON             #
    #-------------------------------------------------#
    def test_parse_json_training_config_with_correct_config(self):
        # test if json training config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'training_config.json')
        ) 
        parsed_config = parse_config_file(args, 'json')
        for key, value in self.training_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_json_prediction_config_with_correct_config(self):
        # test if json prediction config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'prediction_config.json')
        ) 
        parsed_config = parse_config_file(args, 'json')
        for key, value in self.prediction_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_json_evaluation_config_with_correct_config(self):
        # test if json evaluation config is parsed correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'evaluation_config.json')
        ) 
        parsed_config = parse_config_file(args, 'json')
        for key, value in self.evaluation_config.items():
            self.assertEqual(value, parsed_config[key])
    def test_parse_json_base_config_with_wrong_config(self):
        # test if json base config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_base_config.json')
        )
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'json')
        # completely wrong base config should have 2/2 validation errors
        self.assertEqual(2, len(ve.exception.errors()))
    def test_parse_json_training_config_with_wrong_config(self):
        # test if json training config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_training_config.json')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'json')
        # completely wrong training config should have 11/11 validation errors
        self.assertEqual(11, len(ve.exception.errors()))
    def test_parse_json_prediction_config_with_wrong_config(self):
        # test if json prediction config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_prediction_config.json')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'json')
        # completely wrong prediction config should have 6/6 validation errors
        self.assertEqual(6, len(ve.exception.errors()))
    def test_parse_json_evaluation_config_with_wrong_config(self):
        # test if json evaluation config is validated correctly
        args = Namespace(
            config_path=path.join(self.tmp_data.name, 'wrong_evaluation_config.json')
        ) 
        with self.assertRaises(ValidationError) as ve:
            _ = parse_config_file(args, 'json')
        # completely wrong evaluation config should have 4/4 validation errors
        self.assertEqual(4, len(ve.exception.errors()))
#-----------------------------------------------------#
#                 Helper Functions                    #
#-----------------------------------------------------#
def _save_config_to_file(config, file_type, file_path):
    """
    Helper function to save a configuration dictionary to a tempfile during test setup.
    """
    if file_type == 'yml':
        with open(file_path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    else:
        with open(file_path, 'w') as outfile:
            json.dump(config, outfile, indent=4)
def _make_config_erroneous(config, config_type):
    """
    Helper function to make a configuration dictionary erroneous for testing.
    """
    wrong_config = copy.deepcopy(config)
    for key in wrong_config[config_type].keys():
        if type(wrong_config[config_type][key]) is str:
            wrong_config[config_type][key] = 42
        elif wrong_config[config_type][key] is None:
            wrong_config[config_type][key] = 1337
        else:
            wrong_config[config_type][key] = 'a38'
    return wrong_config

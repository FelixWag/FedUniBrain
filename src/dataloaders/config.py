import configparser
import logging

from pathlib import Path


class SingletonMeta(type):
    """
    We only want one instance of the Config class to be created.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    def __init__(self, config_file_name, brats_flag):
        if not hasattr(self, 'is_initialized'):
            self.config = configparser.ConfigParser(allow_no_value=True)
            # Make sure that the keys are not converted to lowercase
            self.config.optionxform = str
            self.config.read(Path('./configs') / config_file_name)
            self.root_path = Path(self.config['data_path']['root_data'])
            self.file_extension = self.config['file_type']['file_extension']
            self._load_settings()
            self._update_multi_seg_paths(brats_flag=brats_flag)
            self.is_initialized = True

    def _load_settings(self):
        self.cropped_input_size = (128, 128, 128)

        self.datasets = {
            'BRATS': {'total_size': 484,
                      'train_size': 444,
                      'channels': ["FLAIR", "T1", "T1c", "T2"],
                      'img_path': self.root_path / self.config['clients_data_path']['BRATS_img'],
                      'seg_path': self.root_path / self.config['clients_data_path']['BRATS_seg'],
                      'filename_img': f"BRATS*_normed_on_mask{self.file_extension}",
                      'filename_seg': f"BRATS*merged{self.file_extension}"},
            'ATLAS': {'total_size': 654,
                      'train_size': 459,
                      'channels': ["T1"],
                      'img_path': self.root_path / self.config['clients_data_path']['ATLAS_img'],
                      'seg_path': self.root_path / self.config['clients_data_path']['ATLAS_seg'],
                      'filename_img': f"*_normed{self.file_extension}",
                      'filename_seg': f"*_label_trimmed{self.file_extension}"},
            'MSSEG': {'total_size': 53,
                      'train_size': 37,
                      'channels': ["FLAIR", "T1", "T1c", "T2", "DP"],
                      'img_path': self.root_path / self.config['clients_data_path']['MSSEG_img'],
                      'seg_path': self.root_path / self.config['clients_data_path']['MSSEG_seg'],
                      'filename_img': f"*{self.file_extension}",
                      'filename_seg': f"*{self.file_extension}"},
            'ISLES': {'total_size': 28,
                      'train_size': 20,
                      'channels': ["FLAIR", "T1", "T2", "DWI"],
                      'img_path': self.root_path / self.config['clients_data_path']['ISLES_img'],
                      'seg_path': self.root_path / self.config['clients_data_path']['ISLES_seg'],
                      'filename_img': f"*{self.file_extension}",
                      'filename_seg': f"*{self.file_extension}"},
            'WMH': {'total_size': 60,
                    'train_size': 42,
                    'channels': ["FLAIR", "T1"],
                    'img_path': self.root_path / self.config['clients_data_path']['WMH_img'],
                    'seg_path': self.root_path / self.config['clients_data_path']['WMH_seg'],
                    'filename_img': f"*{self.file_extension}",
                    'filename_seg': f"*{self.file_extension}"}
        }

    def _update_multi_seg_paths(self, brats_flag):
        if brats_flag:
            logging.info('[IMPORTANT] Updating BRATS seg_path for multi-segmentation')
            self.datasets['BRATS']['seg_path'] = self.root_path / self.config['clients_data_path']['BRATS_multiseg']

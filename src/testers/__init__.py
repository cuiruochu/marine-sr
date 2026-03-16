from .common_tester import CommonTester
from .Bicubic_tester import TesterBicubic


def get_tester(config, model_dict, test_loader):
    if config.model_name.lower() == "bicubic":
        return TesterBicubic(config=config, model_dict=model_dict, test_loader=test_loader)
    return CommonTester(config=config, model_dict=model_dict, test_loader=test_loader)
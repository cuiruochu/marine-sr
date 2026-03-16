from configs import get_test_config, setup_seed
from src.datasets import get_test_loader
from src.models import create_model
from src.testers import get_tester

if __name__ == '__main__':
    setup_seed(0)

    config = get_test_config("configs/test.yaml")

    test_loader = get_test_loader(config)
    model = create_model(config)
    tester = get_tester(config=config, model_dict=model, test_loader=test_loader)
    tester.load_model(config.checkpoint)
    tester.eval()
from tool.test import test_net
from lib.config_file import cfg
from lib.data_build import Data_loader

if __name__ == "__main__":
    #test_net(cfg)
    loader = Data_loader(cfg)
    train_set = loader.construct_loader("test")
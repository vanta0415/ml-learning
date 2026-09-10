# 第7周入口：先训练，再测试

from train import main as train_main
from test import main as test_main

if __name__ == "__main__":
    train_main()
    print("----")
    test_main()
    print("第7周完成")

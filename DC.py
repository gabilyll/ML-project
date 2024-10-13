import random


class DC(object):
    """
    DC细胞类
    """

    def __init__(self, thresholdRange):
        self.csm = 0
        self.semi = 0
        self.mat = 0
        self.count = 0
        self.type = ''
        self.csmth = random.uniform(thresholdRange[0], thresholdRange[1])
        self.antigens = []

    def isMigrate(self):
        """
        是否迁移
        """
        return self.csm > self.csmth

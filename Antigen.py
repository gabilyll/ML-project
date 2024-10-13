class Antigen(object):
    """
    抗原类
    """

    def __init__(self, pamp, ds, ss, ot):
        #
        self.pamp = pamp
        self.ds = ds
        self.ss = ss
        # 实际类型
        self.originalType = int(ot)
        # 采集该抗原的细胞
        self.cells = []
        # 预测类型
        self.type = 0
        self.baise = 1
        # 采集计数
        self.sampleNum = 0

    def tos(self):
        """
        打印必要信息
        """
        print("pamp = ", self.pamp, ", ss = ", self.ss, ", ds = ", self.ds, ', type = ', self.type, ', expect = ', self.originalType)

    def getvalue(self):
        """
        将必要信息格式化成字符串
        """
        return '%f %f %f %d' % (self.ss, self.ds, self.pamp, self.originalType)

    def isCorrect(self):
        """
        判断预测类型与实际类型是否相符
        """
        return self.type == self.originalType

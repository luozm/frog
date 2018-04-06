import configparser


class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version = 'configuration version \'rpn-se-resnext50-fpn\''

        # net
        self.num_classes = 2  # include background class

        # multi-rpn
        self.rpn_base_sizes = [8, 16, 32, 64]  # diameter
        self.rpn_scales = [2,  4,  8,  16]

        aspect = lambda s,x: (s*1/x**0.5,s*x**0.5)

        self.rpn_base_apsect_ratios = [
            [(1,1)],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        ]

        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low = 0.5
#        self.num_train_rpn_anchors = 32000

        self.rpn_train_nms_pre_score_threshold = 0.7
        self.rpn_train_nms_overlap_threshold = 0.8  # higher for more proposals for mask training
        self.rpn_train_nms_min_size = 5

        self.rpn_test_nms_pre_score_threshold = 0.8
        self.rpn_test_nms_overlap_threshold = 0.5
        self.rpn_test_nms_min_size = 5

    # -------------------------------------------------------------------------------------------------------
    def __repr__(self):
        d = self.__dict__.copy()
        str=''
        for k, v in d.items():
            str += '%32s = %s\n' % (k,v)

        return str

    def save(self, file):
        d = self.__dict__.copy()
        config = configparser.ConfigParser()
        config['all'] = d
        with open(file, 'w') as f:
            config.write(f)

    def load(self, file):
        # config = configparser.ConfigParser()
        # config.read(file)
        #
        # d = config['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])

        raise NotImplementedError

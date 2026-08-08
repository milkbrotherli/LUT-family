import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path


class BaseOptions():
    def __init__(self, debug=False):
        self.initialized = False
        self.debug = debug

    def initialize(self, parser):
        parser.add_argument('--debug', default=True, action='store_true')
        parser.add_argument('--validation', default=False, action='store_true')
        parser.add_argument('--transfer', default=False, action='store_true')
        parser.add_argument('--fintune', default=False, action='store_true')
        parser.add_argument('--schedule', default=False, action='store_true')
        parser.add_argument('--testlut', default=False, action='store_true')

        # setting
        parser.add_argument('--model', type=str, default='LUT_ILF_Compact_LUT')
        parser.add_argument('--task',  '-t', type=str, default='in-loop-filter')
        parser.add_argument('--scale', '-r', type=int, default=1, help="up scale factor")
        parser.add_argument('--sigma', '-s', type=int, default=25, help="noise level")
        parser.add_argument('--qf', '-q', type=int, default=20, help="deblocking quality factor")
        parser.add_argument('--nf', type=int, default=64, help="number of filters of convolutional layers")
        parser.add_argument('--stages', type=int, default=4, help="stages of DepthLUT")
        parser.add_argument('--qualityScale', type=int, default=37, help='Quantization Parameter in Codec')
        parser.add_argument('--colorSpace', type=str, default='YUV', help='in YUV or RGB Color Space')
        parser.add_argument('--stage1_modes', type=str, default='sdy', help="sampling modes to use in stage 1")
        parser.add_argument('--stage2_modes', type=str, default='sdy', help="sampling modes to use in stage 2")
        parser.add_argument('--stage3_modes', type=str, default='sdy', help="sampling modes to use in stage 3")
        parser.add_argument('--stage4_modes', type=str, default='sdy', help="sampling modes to use in stage 4")
        parser.add_argument('--weight', action='store_true', default=False, help="template weight is used in every stage")
        parser.add_argument('--interval', type=int, default=4, help='N bit uniform sampling')  # Low_bit_cut_off
        parser.add_argument('--load_from_opt_file', action='store_true', default=False)
        parser.add_argument('--ps_error', action='store_true', default=False)
        parser.add_argument('--conversion', action='store_true', default=True)

        # dir
        parser.add_argument('--logDir', type=str, default='../runs/experiment', help="log folder")
        parser.add_argument('--expDir', type=str, default='../runs/experiment', help="experiment folder")
        parser.add_argument('--loadDir', type=str, default='../checkpoints', help="network checkpoint folder")
        parser.add_argument('--lutDir', type=str, default='../luts/regular', help="input LUT folder")
        parser.add_argument('--lutSaveDir', type=str, default='../luts/finetuned', help="output LUT folder")

        # compact LUT sampling settings
        parser.add_argument('--cd', type=str, default='xy', help='compressed dimensions: xy, xyz, xyzt')
        parser.add_argument('--dw', type=int, default=2, help='diagonal width:2, 3, 4, 5')
        parser.add_argument('--si', type=int, default=5, help='sampling interval of non-diagonal subsampling: 5, 6, 7')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        cli_args = [] if self.debug else sys.argv[1:]
        option_to_dest = {
            option_string: action.dest
            for action in parser._actions
            for option_string in action.option_strings
        }
        self.provided_option_dests = set()
        for argument in cli_args:
            option_string = argument.split('=', 1)[0]
            destination = option_to_dest.get(option_string)
            if destination is not None:
                self.provided_option_dests.add(destination)

        opt = parser.parse_args(cli_args)

        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_options(self, opt):
        file_name = os.path.join(opt.expDir, 'opt')
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def process(self, opt):
        if "dn" in opt.task:
            opt.flag = opt.sigma
        elif "db" in opt.task:
            opt.flag = opt.qf
        elif "sr" in opt.task:
            opt.flag = opt.scale
        else:
            opt.flag = "0"
        return opt

    def save_code(self):
        src_dir = ["../common", "../filter"]
        trg_dir = os.path.join(self.opt.expDir, "code")
        if not os.path.isdir(trg_dir):
            os.mkdir(trg_dir)
        for folder in src_dir:
            for f in Path(folder).rglob("*.py"):
                shutil.copy(f, trg_dir, follow_symlinks=False)

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        opt.isTest = self.isTest
        opt = self.process(opt)

        training_presets = (
            getattr(opt, 'testModel', False),
            getattr(opt, 'trainModel', False),
            getattr(opt, 'resumeModel', False),
        )
        if sum(bool(mode) for mode in training_presets) > 1:
            self.parser.error(
                '--testModel, --trainModel and --resumeModel are mutually exclusive'
            )

        # Keep the historical hard-coded debug configuration for scripts that do
        # not select one of the public training presets. Presets use their normal
        # command-line values so options such as --batchSize remain effective.
        if opt.isTrain and opt.debug and not any(training_presets):
            opt.batchSize = 2
            opt.validPerSize = 20
            opt.validSkipSize = 10
            opt.qualityScale = 37
            opt.cropSize = 8
            opt.trainDir = "../data/milkbrotherli/DIV2K_YUV420"
            opt.trainDir2 = "../data/milkbrotherli/BVI-DVC"
            opt.dataReadDir = "../data/milkbrotherli/dataRead"
            opt.datasetNum = 2
            opt.ps_error = True
            opt.valDir = "../data/milkbrotherli/Benchmark_VVC"
            opt.logDir = "../runs/experiment"
            opt.expDir = "../runs/experiment"
            opt.loadDir = "../checkpoints"
            opt.stage1_modes = "sdy"
            opt.stage2_modes = "sdy"
            opt.stage3_modes = "sdy"
            opt.stage4_modes = "sdy"
            opt.validFast = True
            opt.valA = False
            opt.startIter = 0
            opt.totalIter = 1000000
            opt.displayStep = 10
            opt.valStep = 20
            opt.saveStep = 10
            opt.valStartIter = 0
            opt.nf = 64

        if opt.transfer and opt.debug:
            opt.loadDir = "../checkpoints"
            opt.expDir = r"../luts/regular"
            opt.lutDir = r"../luts/regular"
            opt.lutName = "weight"
            opt.loadIter = 0
            opt.cd = "xyzt"
            opt.dw = 4
            opt.si = 5

        if opt.fintune and opt.debug:
            opt.expDir = "../runs/finetune"
            opt.lutDir = r"../luts/regular"
            opt.lutSaveDir = "../luts/finetuned"
            opt.lutName = "weight"
            opt.lr0 = 1e-3
            opt.lr1 = 1e-4
            opt.loadIter = 0
            opt.startIter = 0
            opt.maxIter = 100000
            opt.valStep = 1000
            opt.saveStep = 1000
            opt.cd = "xyzt"
            opt.dw = 3
            opt.si = 5

        if opt.testlut and opt.debug:
            opt.loadIter = 40000
            opt.lutName = 'weight'
            opt.valDir = "../data/milkbrotherli/Benchmark_VVC"
            opt.lutDir = r"../luts/finetuned"
            opt.testDir = r'../data/milkbrotherli/Benchmark_VVC'
            opt.valA = False

        if opt.isTrain and getattr(opt, 'testModel', False):
            # Full Benchmark evaluation of the bundled historical Y model.
            opt.validation = True
            opt.yModel = 'rfd1'
            opt.ps_error = True
            opt.startIter = opt.testModelIter
            opt.validPerSize = opt.testValidPerSize
            opt.validSkipSize = opt.testValidSkipSize
            opt.validFast = True
            opt.valA = False
            opt.expDir = opt.testModelExpDir
            opt.logDir = opt.testModelExpDir

        if opt.isTrain and getattr(opt, 'trainModel', False):
            # Reproducible public entry point for training the Y model from scratch.
            opt.validation = False
            opt.yModel = 'rfd1'
            opt.ps_error = True
            opt.startIter = 0
            opt.totalIter = opt.trainModelTotalIter
            opt.expDir = opt.trainModelExpDir
            opt.logDir = opt.trainModelExpDir

            # One-iteration, low-memory public verification preset. The full
            # training settings can be supplied explicitly on the command line.
            train_model_defaults = {
                # Data
                'trainDir': '../data/milkbrotherli/DIV2K_YUV420',
                'trainDir2': '../data/milkbrotherli/BVI-DVC',
                'dataReadDir': '../data/milkbrotherli/dataRead',
                'valDir': '../data/milkbrotherli/Benchmark_VVC',
                'datasetNum': 2,
                'qualityScale': 37,
                'colorSpace': 'YUV',

                # Training patches and data loading
                'batchSize': 2,
                'cropSize': 8,
                'workerNum': 0,

                # Network
                'nf': 12,
                'stages': 4,
                'stage1_modes': 'sdy',
                'stage2_modes': 'sdy',
                'stage3_modes': 'sdy',
                'stage4_modes': 'sdy',

                # Optimizer and learning-rate schedule
                'lr0': 1e-3,
                'lr1': 1e-4,
                'weightDecay': 0,
                'steplr': False,
                'Decaystep': 30000,
                'reduce': 0.5,

                # Logging, checkpointing and validation
                'displayStep': 1,
                'valStep': 2000,
                'saveStep': 1,
                'valStartIter': 0,
                'validPerSize': 30,
                'validSkipSize': 3,
                'validFast': False,
                'valA': False,
                'gpuNum': 1,
            }
            for option_name, preset_value in train_model_defaults.items():
                if option_name not in self.provided_option_dests:
                    setattr(opt, option_name, preset_value)

        if opt.isTrain and getattr(opt, 'resumeModel', False):
            # Resume the bundled historical Y checkpoint and its Adam state.
            opt.validation = False
            opt.yModel = 'rfd1'
            opt.ps_error = True
            opt.startIter = opt.resumeModelIter
            opt.totalIter = opt.resumeModelTotalIter
            opt.validFast = True
            opt.valA = False
            opt.expDir = opt.resumeModelExpDir
            opt.logDir = opt.resumeModelExpDir
            opt.loadDir = os.path.dirname(opt.resumeModelPath) or '.'

            # One-iteration, low-memory public resume verification preset. The
            # full resume settings can be supplied explicitly on the command line.
            resume_model_defaults = {
                # Data
                'trainDir': '../data/milkbrotherli/DIV2K_YUV420',
                'trainDir2': '../data/milkbrotherli/BVI-DVC',
                'dataReadDir': '../data/milkbrotherli/dataRead',
                'valDir': '../data/milkbrotherli/Benchmark_VVC',
                'datasetNum': 2,
                'qualityScale': 37,
                'colorSpace': 'YUV',

                # Training patches and data loading
                'batchSize': 2,
                'cropSize': 8,
                'workerNum': 0,

                # Network (must match the resumed checkpoint)
                'nf': 64,
                'stages': 4,
                'stage1_modes': 'sdy',
                'stage2_modes': 'sdy',
                'stage3_modes': 'sdy',
                'stage4_modes': 'sdy',

                # Optimizer and learning-rate schedule
                'lr0': 1e-3,
                'lr1': 1e-4,
                'weightDecay': 0,
                'steplr': False,
                'schedule': False,
                'Decaystep': 30000,
                'reduce': 0.5,

                # Logging, checkpointing and validation
                'displayStep': 1,
                'valStep': 2000,
                'saveStep': 1,
                'valStartIter': 0,
                'validPerSize': 30,
                'validSkipSize': 3,
                'validFast': True,
                'valA': False,
                'gpuNum': 1,
            }
            for option_name, preset_value in resume_model_defaults.items():
                if option_name not in self.provided_option_dests:
                    setattr(opt, option_name, preset_value)

        if opt.isTrain:
            opt.valoutDir = os.path.join(opt.expDir, 'val')
            if not os.path.isdir(opt.valoutDir):
                os.makedirs(opt.valoutDir)
            self.save_options(opt)

        if opt.transfer or opt.fintune:
            if not os.path.isdir(opt.lutDir):
                os.makedirs(opt.lutDir)
            if not os.path.isdir(opt.lutSaveDir):
                os.makedirs(opt.lutSaveDir)
            if not os.path.isdir(opt.expDir):
                os.makedirs(opt.expDir)

        if opt.isTest:
            opt.valoutDir = os.path.join(opt.expDir, 'test')
            if not os.path.isdir(opt.valoutDir):
                os.mkdir(opt.valoutDir)
            self.save_options(opt)

        self.opt = opt

        if not opt.debug:
            self.save_code()

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # data
        parser.add_argument('--cropSize', type=int, default=48, help='input LR training patch size')
        parser.add_argument('--datasetNum', type=int, default=2, help='Total number of training dataset')
        parser.add_argument('--trainDir', type=str, default="../data/milkbrotherli/DIV2K_YUV420")
        parser.add_argument('--trainDir2', type=str, default="../data/milkbrotherli/BVI-DVC")
        parser.add_argument('--dataReadDir', type=str, default="../data/milkbrotherli/dataRead")
        parser.add_argument('--valDir', type=str, default='../data/milkbrotherli/Benchmark_VVC')
        parser.add_argument('--valA', action='store_true', default=False, help='the evaluation of class A')

        # adjust
        parser.add_argument('--batchSize', type=int, default=32)
        parser.add_argument('--validPerSize', type=int, default=30, help='exist frame size of each valid video for val')
        parser.add_argument('--validSkipSize', type=int, default=3, help='valid video frame per skip')
        parser.add_argument('--validFast', action='store_true', default=False, help='valid without large resoluation')
        parser.add_argument('--netFix', action='store_true', default=False, help='the parameters of network are fixed')

        # steplr
        parser.add_argument('--steplr', action='store_true', default=False, help="Learning rate via step decay")
        parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
        parser.add_argument("--Decaystep", type=int, default=30000, help="Learning rate decay every n epochs")

        # training
        parser.add_argument('--lr0', type=float, default=1e-3)
        parser.add_argument('--lr1', type=float, default=1e-4)
        parser.add_argument('--weightDecay', type=float, default=0)

        # adjust
        parser.add_argument('--startIter', type=int, default=0, help='Set 0 for from scratch, else will load saved params and trains further')
        parser.add_argument('--totalIter', type=int, default=300000, help='Total number of training iterations')
        parser.add_argument('--displayStep', type=int, default=100, help='display info every N iteration')
        parser.add_argument('--valStep', type=int, default=2000, help='validate every N iteration')
        parser.add_argument('--saveStep', type=int, default=2000, help='save models every N iteration')
        parser.add_argument('--valStartIter', type=int, default=0, help='start validation at N iteration')
        parser.add_argument('--gpuNum', '-gpu', type=int, default=1)
        parser.add_argument('--workerNum', '-n', type=int, default=8)
        parser.add_argument('--lutName', type=str, default='weight')

        # Full validation using the same evaluation path as training.
        parser.add_argument('--testModel', action='store_true', default=False)
        parser.add_argument(
            '--testModelPath', type=str,
            default='../model-official/net-pretrain/Model_Y_860000.pth'
        )
        parser.add_argument('--testModelIter', type=int, default=860000)
        parser.add_argument('--testValidPerSize', type=int, default=20)
        parser.add_argument('--testValidSkipSize', type=int, default=10)
        parser.add_argument('--testModelExpDir', type=str, default='../runs/model-test-y')

        # Public Y-model training presets. Other training hyperparameters keep the
        # standard options above and can still be overridden on the command line.
        parser.add_argument('--trainModel', action='store_true', default=False)
        parser.add_argument('--trainModelTotalIter', type=int, default=1)
        parser.add_argument('--trainModelExpDir', type=str, default='../runs/train-y')
        parser.add_argument('--resumeModel', action='store_true', default=False)
        parser.add_argument(
            '--resumeModelPath', type=str,
            default='../model-official/net-pretrain/Model_Y_860000.pth'
        )
        parser.add_argument(
            '--resumeOptimizerPath', type=str,
            default='../model-official/net-pretrain/Opt_Y_860000.pth'
        )
        parser.add_argument('--resumeModelIter', type=int, default=860000)
        parser.add_argument('--resumeModelTotalIter', type=int, default=860001)
        parser.add_argument('--resumeModelExpDir', type=str, default='../runs/resume-y')
        parser.add_argument(
            '--yModel', type=str, default='standard',
            choices=('standard', 'rfd1'),
            help='Y network structure used for training/loading'
        )

        # Test after Finetune
        parser.add_argument('--loadIter', type=int, default=0, help='start to load saved LUT')
        parser.add_argument('--maxIter', type=int, default=100000, help='end to load saved LUT')
        parser.add_argument('--validstep', type=int, default=1000)

        self.isTrain = True
        self.isTest = False

        return parser

    def process(self, opt):
        return opt


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--validstep', type=int, default=1000)
        parser.add_argument('--maxIter', type=int, default=100000)
        parser.add_argument('--loadIter', '-i', type=int, default=1000)
        parser.add_argument('--testset', type=str, default='VVC_AI')
        parser.add_argument('--testDir', type=str, default='../data/milkbrotherli/Benchmark_VVC')
        parser.add_argument('--resultRoot', type=str, default='../results')

        parser.add_argument('--lutName', type=str, default='weight')
        parser.add_argument('--totalTest', action='store_true', default=False)
        parser.add_argument('--validFast', action='store_true', default=False, help='valid without large resoluation')
        parser.add_argument('--dataReadDir', type=str, default="../data/milkbrotherli/dataRead")
        parser.add_argument('--valDir', type=str, default='../data/milkbrotherli/Benchmark_VVC')
        parser.add_argument('--validPerSize', type=int, default=30, help='exist frame size of each valid video for val')
        parser.add_argument('--validSkipSize', type=int, default=3, help='valid video frame per skip')
        parser.add_argument('--valA', action='store_true', default=False, help='the evaluation of class A')

        self.isTrain = False
        self.isTest = True

        return parser


class LUTTestOptions:
    """Public Y Regular/Compact LUT evaluation options.

    This parser is intentionally independent from TrainOptions so LUT-only
    evaluation does not inherit training/debug side effects.
    """

    def __init__(self, expected_mode):
        if expected_mode not in ('regular', 'compact'):
            raise ValueError('expected_mode must be regular or compact')
        self.expected_mode = expected_mode
        self.project_root = Path(__file__).resolve().parents[1]
        self.parser = self._build_parser()

    def _build_parser(self):
        description = 'Test the Y-channel {} LUTs'.format(
            self.expected_mode
        )
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        regular_flags = ['--testRegularLUT']
        compact_flags = ['--testCompactLUT']
        if self.expected_mode == 'regular':
            regular_flags.append('--testlut')
        else:
            compact_flags.append('--testlut')
        parser.add_argument(
            *regular_flags, dest='testRegularLUT', action='store_true',
            help='evaluate the public Y regular-LUT preset',
        )
        parser.add_argument(
            *compact_flags, dest='testCompactLUT', action='store_true',
            help='evaluate the public Y compact-LUT preset',
        )

        # LUT and validation paths
        parser.add_argument('--lutDir', default=None)
        parser.add_argument('--valDir', default=None)
        parser.add_argument('--expDir', default=None)

        # LUT structure
        parser.add_argument('--loadIter', type=int, default=None)
        parser.add_argument('--interval', type=int, default=None)
        parser.add_argument('--stage1_modes', default=None)
        parser.add_argument('--stages', type=int, default=None)
        parser.add_argument('--scale', type=int, default=None)
        parser.add_argument('--lutName', default=None)

        # Compact sampling; ignored by the Regular-LUT test
        parser.add_argument(
            '--cd', choices=('xy', 'xyz', 'xyzt'), default=None
        )
        parser.add_argument('--dw', type=int, default=None)
        parser.add_argument('--si', type=int, default=None)

        # Validation
        parser.add_argument('--qualityScale', type=int, default=None)
        parser.add_argument('--validPerSize', type=int, default=None)
        parser.add_argument('--validSkipSize', type=int, default=None)
        parser.add_argument('--fullValidation', action='store_true')

        # Runtime
        parser.add_argument(
            '--device', choices=('auto', 'cuda', 'cpu'), default=None
        )
        return parser

    def _regular_defaults(self):
        # Public Y Regular-LUT test preset. Edit these values to change the
        # release defaults. Explicit command-line values take priority.
        return {
            # LUT
            'lutDir': str(
                self.project_root / 'model-official' / 'regular-lut-pretrain'
            ),
            'lutName': 'weight',
            'loadIter': 46000,
            'interval': 4,

            # RF-1 Y LUT structure
            'stage1_modes': 'sdy',
            'stages': 4,
            'scale': 1,

            # Validation data
            'valDir': str(
                self.project_root / 'data' / 'milkbrotherli'
                / 'Benchmark_VVC'
            ),
            'qualityScale': 37,
            'validPerSize': 1,
            'validSkipSize': 1,

            # Runtime and logging
            'device': 'auto',
            'expDir': str(self.project_root / 'runs' / 'test-regular-y'),
        }

    def _compact_defaults(self):
        # Public Y Compact-LUT test preset. Edit these values to change the
        # release defaults. Explicit command-line values take priority.
        return {
            # LUT
            'lutDir': str(
                self.project_root / 'model-official' / 'compact-lut-pretrain'
                / 'xyzt3i5'
            ),
            'lutName': 'weight',
            'loadIter': 46000,
            'interval': 4,

            # Compact sampling
            'cd': 'xyzt',
            'dw': 3,
            'si': 5,

            # RF-1 Y LUT structure
            'stage1_modes': 'sdy',
            'stages': 4,
            'scale': 1,

            # Validation data
            'valDir': str(
                self.project_root / 'data' / 'milkbrotherli'
                / 'Benchmark_VVC'
            ),
            'qualityScale': 37,
            'validPerSize': 1,
            'validSkipSize': 1,

            # Runtime and logging
            'device': 'auto',
            'expDir': str(self.project_root / 'runs' / 'test-compact-y'),
        }

    def parse(self):
        opt = self.parser.parse_args()
        selected_modes = (
            bool(opt.testRegularLUT), bool(opt.testCompactLUT)
        )
        if sum(selected_modes) != 1:
            self.parser.error(
                'select exactly one of --testRegularLUT and --testCompactLUT'
            )

        selected_mode = (
            'regular' if opt.testRegularLUT else 'compact'
        )
        if selected_mode != self.expected_mode:
            self.parser.error(
                'this entry point requires --test{}LUT'.format(
                    self.expected_mode.capitalize()
                )
            )

        defaults = (
            self._regular_defaults()
            if selected_mode == 'regular'
            else self._compact_defaults()
        )
        for option_name, preset_value in defaults.items():
            if getattr(opt, option_name) is None:
                setattr(opt, option_name, preset_value)
        return opt


class LUTTransferOptions:
    """Public Y network-to-Regular-LUT conversion options."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            description='Convert the bundled Y model to Regular LUTs',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--transferRegularLUT', '--transfer',
            dest='transferRegularLUT', action='store_true',
            help='run the public Y model-to-Regular-LUT preset',
        )

        # Source model and destination
        parser.add_argument('--modelPath', default=None)
        parser.add_argument('--lutDir', default=None)
        parser.add_argument('--recordName', default=None)

        # Network and LUT structure
        parser.add_argument('--loadIter', type=int, default=None)
        parser.add_argument('--nf', type=int, default=None)
        parser.add_argument('--interval', type=int, default=None)
        parser.add_argument('--stage1_modes', default=None)
        parser.add_argument('--stage2_modes', default=None)
        parser.add_argument('--stage3_modes', default=None)
        parser.add_argument('--stage4_modes', default=None)
        parser.add_argument('--stages', type=int, default=None)
        parser.add_argument('--scale', type=int, default=None)
        parser.add_argument('--lutName', default=None)
        parser.add_argument('--ps_error', action='store_true', default=None)

        # Runtime
        parser.add_argument(
            '--device', choices=('auto', 'cuda', 'cpu'), default=None
        )
        parser.add_argument('--transferBatchSize', type=int, default=None)
        return parser

    def _defaults(self):
        # Public Y model-to-Regular-LUT example. Edit these values to change
        # the release defaults. Explicit command-line values take priority.
        return {
            # Source and output
            'modelPath': str(
                self.project_root / 'model-official' / 'net-pretrain'
                / 'Model_Y_860000.pth'
            ),
            'lutDir': str(
                self.project_root / 'LUT-pth' / 'Y-Regular-LUT'
            ),
            'recordName': 'transfer_config.json',
            'loadIter': 860000,

            # Network and LUT structure
            'nf': 64,
            'interval': 4,
            'stage1_modes': 'sdy',
            'stage2_modes': 'sdy',
            'stage3_modes': 'sdy',
            'stage4_modes': 'sdy',
            'stages': 4,
            'scale': 1,
            'lutName': 'weight',
            'ps_error': True,

            # Runtime
            'device': 'auto',
            'transferBatchSize': 2048,
        }

    def parse(self):
        opt = self.parser.parse_args()
        if not opt.transferRegularLUT:
            self.parser.error(
                'run this entry point with --transferRegularLUT'
            )
        for option_name, preset_value in self._defaults().items():
            if getattr(opt, option_name) is None:
                setattr(opt, option_name, preset_value)
        return opt


class LUTCompressionOptions:
    """Public Y Regular-LUT-to-Compact-LUT conversion options."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            description='Convert the bundled Y Regular LUT to Compact LUT',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--transferCompactLUT', action='store_true',
            help='run the public Y Regular-to-Compact LUT preset',
        )

        # Source and destination
        parser.add_argument('--regularLUTDir', default=None)
        parser.add_argument('--compactLUTDir', default=None)
        parser.add_argument('--recordName', default=None)

        # LUT structure and compact sampling
        parser.add_argument('--loadIter', type=int, default=None)
        parser.add_argument('--interval', type=int, default=None)
        parser.add_argument('--cd', choices=('xy', 'xyz', 'xyzt'), default=None)
        parser.add_argument('--dw', type=int, default=None)
        parser.add_argument('--si', type=int, default=None)
        parser.add_argument('--stage1_modes', default=None)
        parser.add_argument('--scale', type=int, default=None)
        parser.add_argument('--lutName', default=None)
        return parser

    def _defaults(self):
        # Public pretrained Y Regular-to-Compact LUT example. Edit these
        # values to change the release defaults. Explicit CLI values win.
        return {
            # Source and output
            'regularLUTDir': str(
                self.project_root / 'model-official' / 'regular-lut-pretrain'
            ),
            'compactLUTDir': str(
                self.project_root / 'LUT-pth' / 'Y-Compact-LUT' / 'xyzt3i5'
            ),
            'recordName': 'compression_config.json',
            'loadIter': 46000,

            # LUT structure
            'interval': 4,
            'stage1_modes': 'sdy',
            'scale': 1,
            'lutName': 'weight',

            # Compact sampling
            'cd': 'xyzt',
            'dw': 3,
            'si': 5,
        }

    def parse(self):
        opt = self.parser.parse_args()
        if not opt.transferCompactLUT:
            self.parser.error(
                'run this entry point with --transferCompactLUT'
            )
        for option_name, preset_value in self._defaults().items():
            if getattr(opt, option_name) is None:
                setattr(opt, option_name, preset_value)
        return opt


class LUTFineTuneOptions:
    """Public pretrained Y Regular-LUT fine-tuning example options."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            description='Fine-tune the bundled Y Regular LUT',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--finetuneRegularLUT', '--fintune',
            dest='finetuneRegularLUT', action='store_true',
            help='run the public pretrained Y Regular-LUT fine-tuning preset',
        )

        # LUT source and output
        parser.add_argument('--inputLUTDir', default=None)
        parser.add_argument('--lutSaveDir', default=None)
        parser.add_argument('--recordName', default=None)
        parser.add_argument('--inputIter', type=int, default=None)

        # LUT structure
        parser.add_argument('--interval', type=int, default=None)
        parser.add_argument('--stage1_modes', default=None)
        parser.add_argument('--stages', type=int, default=None)
        parser.add_argument('--scale', type=int, default=None)
        parser.add_argument('--lutName', default=None)

        # Training data
        parser.add_argument('--trainDir', default=None)
        parser.add_argument('--trainDir2', default=None)
        parser.add_argument('--dataReadDir', default=None)
        parser.add_argument('--datasetNum', type=int, default=None)
        parser.add_argument('--qualityScale', type=int, default=None)
        parser.add_argument('--colorSpace', default=None)
        parser.add_argument('--batchSize', type=int, default=None)
        parser.add_argument('--cropSize', type=int, default=None)
        parser.add_argument('--workerNum', type=int, default=None)

        # Optimizer and example length
        parser.add_argument('--maxIter', type=int, default=None)
        parser.add_argument('--lr0', type=float, default=None)
        parser.add_argument('--lr1', type=float, default=None)
        parser.add_argument('--weightDecay', type=float, default=None)
        parser.add_argument('--displayStep', type=int, default=None)
        parser.add_argument('--saveStep', type=int, default=None)
        parser.add_argument('--seed', type=int, default=None)

        # Runtime
        parser.add_argument(
            '--device', choices=('auto', 'cuda', 'cpu'), default=None
        )
        return parser

    def _defaults(self):
        # Public fine-tuning example. Increase maxIter and saveStep on the
        # command line for a full experiment.
        return {
            # LUT source and output
            'inputLUTDir': str(
                self.project_root / 'model-official' / 'regular-lut-pretrain'
            ),
            'lutSaveDir': str(
                self.project_root / 'LUT-pth'
                / 'Y-Regular-LUT-Finetuned'
            ),
            'recordName': 'finetune_config.json',
            'inputIter': 46000,

            # RF-1 Y LUT structure
            'interval': 4,
            'stage1_modes': 'sdy',
            'stages': 4,
            'scale': 1,
            'lutName': 'weight',

            # Bundled training data
            'trainDir': str(
                self.project_root / 'data' / 'milkbrotherli'
                / 'DIV2K_YUV420'
            ),
            'trainDir2': str(
                self.project_root / 'data' / 'milkbrotherli' / 'BVI-DVC'
            ),
            'dataReadDir': str(
                self.project_root / 'data' / 'milkbrotherli' / 'dataRead'
            ),
            'datasetNum': 1,
            'qualityScale': 37,
            'colorSpace': 'YUV',
            'batchSize': 2,
            'cropSize': 8,
            'workerNum': 0,

            # Optimizer and short example length
            'maxIter': 1,
            'lr0': 1e-3,
            'lr1': 1e-4,
            'weightDecay': 0.0,
            'displayStep': 1,
            'saveStep': 1,
            'seed': 1234,

            # Runtime
            'device': 'auto',
        }

    def parse(self):
        opt = self.parser.parse_args()
        if not opt.finetuneRegularLUT:
            self.parser.error(
                'run this entry point with --finetuneRegularLUT'
            )
        for option_name, preset_value in self._defaults().items():
            if getattr(opt, option_name) is None:
                setattr(opt, option_name, preset_value)
        if opt.maxIter < 1:
            self.parser.error('--maxIter must be at least 1')
        if opt.lr0 <= 0 or opt.lr1 <= 0:
            self.parser.error('--lr0 and --lr1 must be positive')
        if opt.lr1 > opt.lr0:
            self.parser.error('--lr1 must not be greater than --lr0')
        if opt.saveStep < 1 or opt.displayStep < 1:
            self.parser.error('--saveStep and --displayStep must be positive')
        return opt


class CompactLUTFineTuneOptions:
    """Public pretrained Y Compact-LUT fine-tuning example options."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            description='Fine-tune the bundled Y Compact LUT',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--finetuneCompactLUT', '--fintune',
            dest='finetuneCompactLUT', action='store_true',
            help='run the public pretrained Y Compact-LUT fine-tuning preset',
        )

        # LUT source and output
        parser.add_argument('--inputLUTDir', default=None)
        parser.add_argument('--lutSaveDir', default=None)
        parser.add_argument('--recordName', default=None)
        parser.add_argument('--inputIter', type=int, default=None)

        # LUT structure and Compact sampling
        parser.add_argument('--interval', type=int, default=None)
        parser.add_argument('--cd', choices=('xy', 'xyz', 'xyzt'), default=None)
        parser.add_argument('--dw', type=int, default=None)
        parser.add_argument('--si', type=int, default=None)
        parser.add_argument('--stage1_modes', default=None)
        parser.add_argument('--stages', type=int, default=None)
        parser.add_argument('--scale', type=int, default=None)
        parser.add_argument('--lutName', default=None)

        # Training data
        parser.add_argument('--trainDir', default=None)
        parser.add_argument('--trainDir2', default=None)
        parser.add_argument('--dataReadDir', default=None)
        parser.add_argument('--datasetNum', type=int, default=None)
        parser.add_argument('--qualityScale', type=int, default=None)
        parser.add_argument('--colorSpace', default=None)
        parser.add_argument('--batchSize', type=int, default=None)
        parser.add_argument('--cropSize', type=int, default=None)
        parser.add_argument('--workerNum', type=int, default=None)

        # Optimizer and example length
        parser.add_argument('--maxIter', type=int, default=None)
        parser.add_argument('--lr0', type=float, default=None)
        parser.add_argument('--weightDecay', type=float, default=None)
        parser.add_argument('--displayStep', type=int, default=None)
        parser.add_argument('--saveStep', type=int, default=None)
        parser.add_argument('--seed', type=int, default=None)

        # Runtime
        parser.add_argument(
            '--device', choices=('auto', 'cuda', 'cpu'), default=None
        )
        return parser

    def _defaults(self):
        # One-iteration public Compact-LUT fine-tuning example. Increase
        # maxIter/saveStep for a full experiment.
        return {
            # LUT source and output
            'inputLUTDir': str(
                self.project_root / 'model-official' / 'compact-lut-pretrain'
                / 'xyzt3i5'
            ),
            'lutSaveDir': str(
                self.project_root / 'LUT-pth'
                / 'Y-Compact-LUT-Finetuned' / 'xyzt3i5'
            ),
            'recordName': 'finetune_config.json',
            'inputIter': 46000,

            # RF-1 Y Compact-LUT structure
            'interval': 4,
            'cd': 'xyzt',
            'dw': 3,
            'si': 5,
            'stage1_modes': 'sdy',
            'stages': 4,
            'scale': 1,
            'lutName': 'weight',

            # Bundled training data
            'trainDir': str(
                self.project_root / 'data' / 'milkbrotherli'
                / 'DIV2K_YUV420'
            ),
            'trainDir2': str(
                self.project_root / 'data' / 'milkbrotherli' / 'BVI-DVC'
            ),
            'dataReadDir': str(
                self.project_root / 'data' / 'milkbrotherli' / 'dataRead'
            ),
            'datasetNum': 1,
            'qualityScale': 37,
            'colorSpace': 'YUV',
            'batchSize': 2,
            'cropSize': 8,
            'workerNum': 0,

            # Optimizer and short example length
            'maxIter': 1,
            'lr0': 1e-3,
            'weightDecay': 0.0,
            'displayStep': 1,
            'saveStep': 1,
            'seed': 1234,

            # Runtime
            'device': 'auto',
        }

    def parse(self):
        opt = self.parser.parse_args()
        if not opt.finetuneCompactLUT:
            self.parser.error(
                'run this entry point with --finetuneCompactLUT'
            )
        for option_name, preset_value in self._defaults().items():
            if getattr(opt, option_name) is None:
                setattr(opt, option_name, preset_value)
        if opt.maxIter < 1:
            self.parser.error('--maxIter must be at least 1')
        if opt.saveStep < 1 or opt.displayStep < 1:
            self.parser.error('--saveStep and --displayStep must be positive')
        return opt

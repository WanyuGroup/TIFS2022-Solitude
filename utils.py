import os
import time
from argparse import ArgumentTypeError, Action
import inspect
import enum
import functools
from subprocess import check_call, DEVNULL, STDOUT
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def measure_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print(f'\nTotal time spent in {str(func.__name__)}:', end - start, 'seconds.\n\n')
        return out

    return wrapper


class WandbLogger:
    def __init__(self, project=None, name=None, config=None, save_code=True,
                 reinit=True, enabled=True, **kwargs):
        self.enabled = enabled
        if enabled:
            if wandb is None:
                raise ImportError('wandb is not installed yet, install it with `pip install wandb`.')

            os.environ["WANDB_SILENT"] = "true"

            settings = wandb.Settings(start_method="fork")  # noqa

            self.experiment = wandb.init(
                name=name, project=project,
                reinit=reinit, resume='allow', config=config, save_code=save_code,
                settings=settings,
                **kwargs)

    def log(self, metrics):
        if self.enabled:
            self.experiment.log(metrics)

    def log_summary(self, metrics):
        if self.enabled:
            for metric, value in metrics.items():
                self.experiment.summary[metric] = value

    def watch(self, model):
        if self.enabled:
            self.experiment.watch(model, log_freq=50)

    def finish(self):
        if self.enabled:
            self.experiment.finish()


class JobManager:
    def __init__(self, args, cmd_generator=None):
        self.args = args
        self.name = args.name
        self.command = args.command
        self.jobs_dir = args.jobs_dir
        self.cmd_generator = cmd_generator

    def run(self):
        if self.command == 'create':
            self.create()
        elif self.command == 'submit':
            self.submit()
        elif self.command == 'status':
            self.status()
        elif self.command == 'resubmit':
            self.resubmit()
        elif self.command == 'exec':
            self.exec()

    def create(self):
        os.makedirs(self.jobs_dir, exist_ok=True)
        run_cmds = self.cmd_generator(self.args)

        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs'), 'w') as file:
            for run in tqdm(run_cmds):
                file.write(run + '\n')

        print('job file created:', os.path.join(self.jobs_dir, f'{self.name}.jobs'))

    def submit(self):
        window = 7500
        num_cmds = sum(1 for _ in open(os.path.join(self.jobs_dir, f'{self.name}.jobs')))

        for i in tqdm(range(0, num_cmds, window), desc='submitting jobs'):
            begin = i + 1
            end = min(i + window, num_cmds)

            job_file_content = [
                f'#$ -N {self.name}-{begin}-{end}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P dusk2dawn\n',
                f'#$ -l pytorch,sgpu,gpumem=10\n',
                f'#$ -t {begin}-{end}\n',
                f'#$ -o {self.jobs_dir}\n',
                f'#$ -e {self.jobs_dir}\n',
                f'#$ -cwd\n',
                f'#$ -V\n',
                f'python experiments.py -n {self.name} exec --id $SGE_TASK_ID \n'
            ]

            file_name = os.path.join(self.jobs_dir, f'{self.name}-{begin}-{end}.job')

            with open(file_name, 'w') as file:
                file.writelines(job_file_content)
                file.flush()

            check_call(['qsub', file_name], stdout=DEVNULL, stderr=STDOUT)

        print('done')

    def resubmit(self):
        failed_jobs = self.get_failed_jobs()

        if len(failed_jobs):
            with open(os.path.join(self.jobs_dir, f'{self.name}.jobs')) as jobs_file:
                job_list = jobs_file.read().splitlines()

            self.cmd_generator = lambda args: [job_list[i-1] for i,_,_ in failed_jobs]
            self.name = f'{self.name}-resubmit'
            self.create()
            self.submit()


    def status(self):
        try:
            import tabulate
        except ImportError:
            tabulate = None

        failed_jobs = self.get_failed_jobs()

        if tabulate:
            print(tabulate.tabulate(failed_jobs, headers=['job id', 'error file', 'num lines']))
        else:
            for _, file, num_lines in failed_jobs:
                print(num_lines, os.path.join(self.jobs_dir, file))

        print()

    def exec(self):
        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs')) as jobs_file:
            job_list = jobs_file.read().splitlines()

        if self.args.all:
            for cmd in job_list:
                check_call(cmd.split())
        else:
            check_call(job_list[self.args.id-1].split())

    def get_failed_jobs(self):
        failed_jobs = []
        file_list = [
            os.path.join(self.jobs_dir, file)
            for file in os.listdir(self.jobs_dir) if file.startswith(self.name) and file.count('.e')
        ]

        for file in file_list:
            num_lines = sum(1 for _ in open(file))
            if num_lines > 0:
                job_id = int(file.split('.')[-1])
                failed_jobs.append([job_id, file, num_lines])

        return failed_jobs

    @staticmethod
    def register_arguments(parser, default_jobs_dir='./jobs'):
        parser.add_argument('-n', '--name', type=str, required=True)
        parser.add_argument('-j', '--jobs-dir', type=str, default=default_jobs_dir)
        command_subparser = parser.add_subparsers(dest='command')

        parser_create = command_subparser.add_parser('create')
        command_subparser.add_parser('submit')
        command_subparser.add_parser('status')
        command_subparser.add_parser('resubmit')

        parser_exec = command_subparser.add_parser('exec')
        parser_exec.add_argument('--id', type=int)
        parser_exec.add_argument('--all', action='store_true')

        return parser, parser_create


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def add_parameters_as_argument(function, parser):
    if inspect.isclass(function):
        function = function.__init__
    parameters = inspect.signature(function).parameters
    for param_name, param_obj in parameters.items():
        if param_obj.annotation is not inspect.Parameter.empty:
            arg_info = param_obj.annotation
            arg_info['default'] = param_obj.default
            arg_info['dest'] = param_name
            arg_info['type'] = arg_info.get('type', type(param_obj.default))

            if arg_info['type'] is bool:
                arg_info['type'] = str2bool
                arg_info['nargs'] = '?'
                arg_info['const'] = True

            if 'choices' in arg_info:
                arg_info['help'] = arg_info.get('help', '') + f" (choices: {', '.join(arg_info['choices'])})"
                arg_info['metavar'] = param_name.upper()

            options = {f'--{param_name}', f'--{param_name.replace("_", "-")}'}
            custom_options = arg_info.pop('option', [])
            custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
            options.update(custom_options)
            options = sorted(sorted(list(options)), key=len)
            parser.add_argument(*options, **arg_info)


def strip_unexpected_kwargs(func, kwargs):
    signature = inspect.signature(func)
    parameters = signature.parameters

    # check if the function has kwargs
    for name, param in parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
    return kwargs


def from_args(func, ns, *args, **kwargs):
    return func(*args, **strip_unexpected_kwargs(func, vars(ns)), **kwargs)


def print_args(args):
    message = [f'{name}: {colored_text(str(value), TermColors.FG.cyan)}' for name, value in vars(args).items()]
    print(', '.join(message) + '\n')


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color.value + msg + TermColors.Control.reset.value


class Enum(enum.Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class EnumAction(Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        _enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if _enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(_enum, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in _enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = _enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        enum = self._enum(values)  # noqa
        setattr(namespace, self.dest, enum)


class TermColors:
    class Control(enum.Enum):
        reset = '\033[0m'
        bold = '\033[01m'
        disable = '\033[02m'
        underline = '\033[04m'
        reverse = '\033[07m'
        strikethrough = '\033[09m'
        invisible = '\033[08m'

    class FG(enum.Enum):
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class BG(enum.Enum):
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'

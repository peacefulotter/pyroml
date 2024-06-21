from torch.optim import Adam


class Config:

    def __init__(
        self,
        name,
        lr=1e-4,
        max_epochs=1,
        max_iterations=None,
        optimizer=Adam,
        optimizer_params=None,
        scheduler=None,
        scheduler_params=None,
        grad_norm_clip=1.0,
        evaluate=True,
        evaluate_every=10,
        eval_max_iterations=None,
        device="auto",
        compile=True,
        checkpoint_folder="./checkpoints",
        batch_size=64,
        eval_batch_size=None,
        num_workers=4,
        eval_num_workers=0,
        wandb=True,
        wandb_project=None,
        verbose=False,
        debug=False,
    ):
        """
        Configuration object with the specified hyperparameters.

        Args:
            name (str): Name of the configuration.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            max_epochs (int, optional): Number of epochs (if max_iterations is not defined). Defaults to 1.
            max_iterations (int): Maximum number of iterations. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to AdamW.
            optimizer_params (dict, optional): Optimizer parameters. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler. Defaults to None.
            scheduler_params (dict, optional): Scheduler parameters. Defaults to None.
            grad_norm_clip (float, optional): Gradient norm clipping. Defaults to 1.0.
            evaluate (bool or str, optional): Whether to periodically evaluate the model on the evaluation dataset, or 'epoch' to evaluate every epoch. Defaults to True.
            evaluate_every (int, optional): Evaluate every `evaluate_every` iterations / or epoch if evaluate is set to 'epoch'. Defaults to 10.
            eval_max_iterations (int, optional): Maximum number of iterations for the evaluation dataset. Defaults to None.
            device (str, optional): Device to train on. Defaults to "auto" which will use GPU if available.
            compile (bool, optional): Whether to compile the model, this can significantly improve training time but is not supported on all GPUs. Defaults to True.
            checkpoint_folder (str, optional): Folder to save checkpoints. Defaults to "./checkpoints".
            batch_size (int, optional): Batch size. Defaults to 64.
            eval_batch_size (int, optional): Batch size for the evaluation dataset. Defaults to None in which case it will be equal to the training batch size.
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 4.
            eval_num_workers (int, optional): Number of workers for the evaluation dataloader. Note that a value > 0 can cause an AssertionError: 'can only test a child process during evaluation'. Defaults to 0.
            wandb (bool, optional): Whether to use wandb. Defaults to True.
            wandb_project (str, optional): Wandb project name, if wandb is set to True. Defaults to None.
            verbose (bool, optional): Whether to print details of whats going on in the system. Defaults to False.
            debug: (bool, optional): Whether to print debug information. Defaults to False.
        Returns:
            Config: Configuration object with the specified hyperparameters.
        """

        scheduler_params = scheduler_params or {}
        optimizer_params = optimizer_params or {}
        eval_batch_size = eval_batch_size or batch_size

        self.name = name

        # Training
        self.lr = lr
        self.max_epochs = max_epochs
        self.max_iterations = max_iterations
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.grad_norm_clip = grad_norm_clip

        # Validation
        self.evaluate = evaluate
        self.evaluate_every = evaluate_every
        self.eval_max_iterations = eval_max_iterations

        # Model
        self.device = device
        self.compile = compile
        self.checkpoint_folder = checkpoint_folder

        # Data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers

        # Logging
        self.wandb = wandb
        self.wandb_project = wandb_project
        self.verbose = verbose
        self.debug = debug

    def __str__(self):
        return f"Config({str(self.__dict__)[1:-1]})"

    def __repr__(self):
        return self.__str__()

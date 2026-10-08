import numpy as np
import torch, random
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from typing import Any

from accelerate import Accelerator
from argparse import Namespace

plt.switch_backend('agg')


def adjust_learning_rate(
        accelerator: Accelerator | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        epoch: int | None = None,
        args: Namespace | None = None,
        printout: bool = True
) -> None:
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        """
        Identify when to save training progress, and save it when conditions are met.

        Stop training if the loss has not been improving for a number of epochs defined by patience, and
        save progress when there is improvement if requested. An improvement means the loss has improved,
        i.e, decreased by at least delta.

        :param accelerator: accelerator
        :param patience: number of epochs to wait for improvement before stopping training
        :param verbose: whether to print number of epochs without improvement and saving notice.
        :param delta: minimal acceptable improvement of the loss function.
        :param save_mode: whether to save checkpoints or not.
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self,
                 val_loss: float,
                 model: torch.nn.Module,
                 path: Path | str,
                 optimizer: torch.optim.Optimizer | None = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
        """
        Compare current score with best score and update early stopping counter.

        The score is defined as the opposite of the loss. We want to minimize the losse function, hence,
        we want to maximize the loss. If the best score is not defined yet, we are likely just done with
        the first epoch; this will be our first checkpoint. If the score has not significantly improved,
        we increase the early stopping counter. Otherwise, we update the best loss to the current loss
        and save another checkpoint.

        :param val_loss: current validation loss
        :param model: trained neural network
        :param path: path to save the checkpoint
        :param optimizer: optimizer, optional
        :param scheduler: learning rate scheduler, optional
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path, optimizer, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path, optimizer, scheduler)
            self.counter = 0

    def save_checkpoint(self,
                        val_loss: float,
                        model: torch.nn.Module,
                        path: str | Path,
                        optimizer: torch.optim.Optimizer | None = None,
                        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
                        ) -> None:
        """
        Save a general checkpoint for inference and/or resuming training.

        The checkpoint is a dictionnary that contains all information about the state
        of the model, its optimizer and learning rate scheduler.

        :param val_loss: current validation loss
        :param model: trained neural network
        :param path: path to save the checkpoint
        :param optimizer: optimizer, optional
        :param scheduler: learning rate scheduler, optional
        """
        if self.verbose:
            save_msg = (f'Validation loss decreased ({self.val_loss_min:.6f} --> '
                        f'{val_loss:.6f}).  Saving model ...')
            if self.accelerator is not None:
                self.accelerator.print(save_msg)
            else:
                print(save_msg)

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)

        model_dict = create_checkpoint_dict(model, val_loss, optimizer, scheduler)
        path = Path(path)
        if path.is_dir():
            # For retro-compatibility with original Time-LLM
            path = path / 'checkpoint.pth'
        elif path.suffix == '':
            path = path / '.pth'
        torch.save(model_dict, path)

        self.val_loss_min = val_loss


def create_checkpoint_dict(model: torch.nn.Module,
                           loss: float,
                           epoch: int | None = None,
                           optimizer: torch.optim.Optimizer | None = None,
                           scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
                           ) -> dict[str, Any]:
    """
    Create a general checkpoint for inference and/or resuming training.

    The checkpoint is a disctionnary that contains the trained weights of the model,
    and, optionally, the state of the optimizer and of the learning rate scheduler.
    To be used with save_checkpoint of class EarlyStopping.
    """
    checkpoint = {
        'model': model.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    return checkpoint


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def vali_pulsar(args, accelerator, model, discr, vali_data, vali_loader, criterion):
    total_loss = []
    total_loss_d = []
    model.eval()
    discr.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred_dlabels = discr(outputs).detach()
            true_dlabels = discr(batch_y).detach()

            loss = criterion(pred_dlabels, torch.full(pred_dlabels.shape, 1.0, device=accelerator.device))
            loss_d = criterion(pred_dlabels, torch.full(
                pred_dlabels.shape, 0.0, device=accelerator.device)) + criterion(
                true_dlabels, torch.full(pred_dlabels.shape, 1.0, device=accelerator.device))

            total_loss.append(loss.item())
            total_loss_d.append(loss_d.item())

    total_loss = np.average(total_loss)
    total_loss_d = np.average(total_loss_d)

    model.train()
    discr.eval()
    return total_loss, total_loss_d, pred_dlabels.mean().cpu(), true_dlabels.mean().cpu()


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args: Namespace, *, prompt_bank: Path | None = None):
    """
    Load the content of a prompt file.

    `file` may be either the absolute path to a prompt file on disk, or it
    may be a data type specifier like 'ETT'. In the latter case, the
    prompt file will attempt to be loaded from the `prompt_bank` directory,
    which defaults to `datasets/prompt_bank` under `timellm`'s install
    directory if not set.

    Args:
        args: Namespace containing, under .dat, prompt file or data specifier
            for prompt file, to load.
        prompt_bank: The location on disk of the directory in which prompt
            files are stored. Defaults to the `timellm` install directory.

    Returns:
        Content of the prompt file.
    """

    # Ensures compatibilty with orginal Time-LLM
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    file = Path(file)
    if not file.suffix:
        file = file.with_suffix('.txt')

    if prompt_bank is None:
        # Safer than importlib.resources.files
        prompt_bank = Path(__file__).parent.parent.parent.parent / "dataset" / "prompt_bank"

    if file.exists():
        file_location = file
    elif (prompt_bank / file).exists():
        file_location = prompt_bank / file
    else:
        raise OSError(f"{file} not found")

    with file_location.open("r") as f:
        return f.read()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
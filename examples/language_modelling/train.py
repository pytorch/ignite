import logging
from argparse import ArgumentParser
from typing import Tuple

import torch
import torch as th
from torch import nn
from torch.nn import functional as F, AdaptiveLogSoftmaxWithLoss

from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import CategoricalAccuracy

from data import load_data, LMLoader
from model import RNNModel
from utils import set_weight, detach, Average, RunningAverage

logger = logging.getLogger(__name__)


def main(path: str,
         nn_type: str,
         ninp: int,
         nhid: int,
         nlayers: int,
         dropout=0.5,
         dropouth=0.5,
         dropouti=0.5,
         dropoute=0.5,
         wdrop=0.0,
         tie_weights=True,
         bsz=64,
         bptt=70,
         max_epochs=10,
         eval_bsz=None,
         eval_bptt=None,
         lr=0.01,
         clip=0.0,
         adaptive=False,
         cutoffs=None,
         device='cuda',
         to_device=False,
         ckpt_dir='/tmp/',
         ckpt_prefix='LM-model',
         print_freq=100,
         validation_freq=100,
         seed=0):

    torch.manual_seed(seed)
    logger.debug(f'Set manual seed to: {seed}. Creating data loaders...')

    eval_bsz = eval_bsz or bsz
    eval_bptt = eval_bptt or bptt

    (train,
     dev,
     _) = load_data(path)
    trn_dset = LMLoader(train, device=device, bptt=bptt, batch_size=bsz,
                        evaluation=False, to_device=to_device)
    val_dset = LMLoader(dev, device=device, bptt=eval_bptt, batch_size=eval_bsz,
                        evaluation=True, to_device=to_device)

    ntokens = max(int(train.max()), int(dev.max())) + 1
    nout = ninp if tie_weights else nhid

    logger.debug(f'Dataset lengths: '
                 f'{len(trn_dset)}, {len(val_dset)}. '
                 f'Ntokens: {ntokens}. '
                 f'Creating model...')

    module = RNNModel(
        rnn_type=nn_type, ntoken=ntokens, ninp=ninp, nhid=nhid,
        nlayers=nlayers,
        dropout=dropout, dropouth=dropouth, dropouti=dropouti,
        dropoute=dropoute,
        wdrop=wdrop, tie_weights=tie_weights,
    )

    if adaptive:
        logger.debug(f'Creating adaptive decoder for cutoffs {cutoffs}...')
        decoder = AdaptiveLogSoftmaxWithLoss(in_features=nout,
                                             n_classes=ntokens,
                                             cutoffs=cutoffs,
                                             head_bias=(not tie_weights))
        loss_fn = lambda input, target: decoder(input, target).loss

    else:

        logger.debug('Creating standard decoder...')
        decoder = nn.Linear(in_features=nout, out_features=ntokens,
                            bias=(not tie_weights))
        loss_fn = lambda input, target: F.cross_entropy(decoder(input), target)

    if tie_weights:
        set_weight(decoder, module.encoder.weight)

    def traininig_step(engine: Engine, batch: Tuple[th.Tensor, th.Tensor]):
        x, y = batch  # x: (L, B);  y: (L, )

        module.train()
        decoder.train()
        optimizer.zero_grad()
        engine.state.hidden = detach(engine.state.hidden)

        (output,  # (L, B, n_hidden) or (L, B, n_input)
         engine.state.hidden) = module(x, engine.state.hidden)

        output = output.view(-1, output.size(2))

        loss = loss_fn(output, y)  # type: th.Tensor
        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        optimizer.step()
        trn_loss.update(loss.item())

        # saving that tiny bit of memory
        engine.state.batch = None
        engine.state.output = None

    def validation_step(engine: Engine, batch: Tuple[th.Tensor, th.Tensor]):
        with th.no_grad():
            x, y = batch

            module.eval()
            decoder.eval()
            engine.state.hidden = detach(engine.state.hidden)

            (output,
             engine.state.hidden) = module(x, engine.state.hidden)

            output = output.view(-1, output.size(2))
            n_items = output.shape[0] * 1.0

            if adaptive:
                output = decoder.log_prob(output)
                loss = F.nll_loss(output, y)
            else:
                output = decoder(output)
                loss = F.cross_entropy(output, y)

            val_loss.update(loss.item(), count=n_items)
            accuarcy.update((output, y))

            # saving that tiny bit of memory
            engine.state.batch = None
            engine.state.output = None

    module = module.to(device)
    decoder = decoder.to(device)
    params = list(module.parameters()) + list(decoder.parameters())
    optimizer = th.optim.Adam(params, lr=lr, amsgrad=True)
    trainer = Engine(traininig_step)
    validator = Engine(validation_step)
    timer = Timer().attach(trainer)
    trn_loss = RunningAverage(0.99)
    val_loss = Average()
    accuarcy = CategoricalAccuracy()
    chkpointer = ModelCheckpoint(ckpt_dir, ckpt_prefix,
                                 save_interval=1, n_saved=10,
                                 require_empty=False)

    logger.debug(f'Helpers are ready...')

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer_loss_hidden(engine: Engine):
        timer.reset()
        trn_loss.reset()

        engine.state.hidden = module.init_hidden(bsz)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_loss(engine: Engine):
        if (engine.state.iteration % print_freq) == 0:
            it = engine.state.iteration
            ls = trn_loss.compute()
            msg = f'Iteration no {it} / {len(trn_dset)}, loss: {ls}'
            logger.debug(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine: Engine):
        engine.state.batch = None
        engine.state.hidden = None
        engine.state.output = None

        th.cuda.empty_cache()

        chkpointer(engine, {
            'module': module,
            'decoder': decoder
        })

        engine.state.chkpointer = chkpointer

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        ts = round(timer.value() / 60., 2)
        loss = trn_loss.compute()

        logger.info(
            f'Epoch {engine.state.epoch} done. | '
            f'Time elapsed: {ts:.3f}[min] | '
            f'Average loss: {loss} | '
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine: Engine):
        if engine.state.epoch % validation_freq == 0:
            validator.run(val_dset)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine: Engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            chkpointer(engine, {
                'module': module,
                'decoder': decoder
            })

            engine.terminate()

        else:
            raise e

    @validator.on(Events.EPOCH_STARTED)
    def setup(engine: Engine):
        engine.state.batch = None
        engine.state.hidden = None
        engine.state.output = None

        th.cuda.empty_cache()

        val_loss.reset()
        accuarcy.reset()

        engine.state.hidden = module.init_hidden(eval_bsz)

    @validator.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        ls = engine.state.metrics['loss'] = val_loss.compute()
        ac = engine.state.metrics['accuracy'] = accuarcy.compute()

        logger.info(
            f'Validation done. | '
            f'Average loss: {ls:.2f} | '
            f'Average accuracy: {ac:.2f} | '
        )

    logger.debug(f'Invoking trainer.run for {max_epochs} epochs')
    return trainer.run(data=trn_dset, max_epochs=max_epochs)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, required=True,
                        help='')
    parser.add_argument('--nn_type', type=str, default='LSTM',
                        help='')
    parser.add_argument('--ninp', type=int, default=128,
                        help='')
    parser.add_argument('--nhid', type=int, default=256,
                        help='')
    parser.add_argument('--nlayers', type=float, default=2,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='')
    parser.add_argument('--dropouth', type=float, default=0.5,
                        help='')
    parser.add_argument('--dropouti', type=float, default=0.5,
                        help='')
    parser.add_argument('--dropoute', type=float, default=0.5,
                        help='')
    parser.add_argument('--wdrop', type=float, default=0.0,
                        help='')
    parser.add_argument('--tie_weights', action='store_true',
                        help='')
    parser.add_argument('--bsz', type=int, default=32,
                        help='')
    parser.add_argument('--bptt', type=int, default=70,
                        help='')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='')
    parser.add_argument('--eval_bsz', type=int, default=10,
                        help='')
    parser.add_argument('--eval_bptt', type=int, default=70,
                        help='')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='')
    parser.add_argument('--clip', type=float, default=0.0,
                        help='')
    parser.add_argument('--adaptive', action='store_true',
                        help='')
    parser.add_argument('--cutoffs', type=int, nargs='+',
                        help='')
    parser.add_argument('--device', type=str, default='cuda',
                        help='')
    parser.add_argument('--to_device', action='store_true',
                        help='')
    parser.add_argument('--ckpt_dir', type=str, default='.',
                        help='')
    parser.add_argument('--ckpt_prefix', type=str, default='LM-model',
                        help='')
    parser.add_argument('--validation_freq', type=int, default=100,
                        help='')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='')
    parser.add_argument('--seed', type=int, default=42,
                        help='')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.__dict__.pop('debug'):
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt, level=level)
    logging.getLogger('ignite').setLevel(logging.INFO)

    main(**args.__dict__)

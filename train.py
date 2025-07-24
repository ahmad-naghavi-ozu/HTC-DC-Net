import torch
import argparse
import wandb
import os
from datetime import datetime
import time
from tqdm import tqdm

from utils import AverageMeter, UpdatableDict, convert_from_string, data_to_device, load_yaml, \
    fix_seed_for_reproducability, save_config
from build import get_model_and_optimizer
from dataloaders import get_train_val_dataloaders
from cuda_setup import setup_cuda

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config', default=None, help='Specify a config file path')
    parser.add_argument('--exp_config', default=None, help='Specify an experiment config file path')
    parser.add_argument('--restore', action='store_true', help='Restore the run')
    parser.add_argument('--overfit', action='store_true', help='Overfit on small batches for debugging')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--gpu_ids', type=str, default="0", help='GPU IDs to use (comma-separated, e.g., "0,1,2")')
    args, unknown_raw = parser.parse_known_args()
    unknown = []
    for ur in unknown_raw:
        unknown.extend(ur.split("="))
    return args, unknown

class DummyLogger:
    """A dummy logger to use when wandb is disabled"""
    def __init__(self):
        self.id = "dummy_run"
        print("Using dummy logger (wandb disabled)")
    
    def log(self, data):
        # Print only a subset of metrics for clarity
        if 'epoch' in data:
            metrics_str = []
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    metrics_str.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            print(f"Epoch {data.get('epoch')}: " + ", ".join(metrics_str[:5]) + ("..." if len(metrics_str) > 5 else ""))
    
    def watch(self, model):
        pass
    
    def config(self):
        return DummyConfig()

class DummyConfig:
    def update(self, *args, **kwargs):
        pass

def train(cfgs, logger, train_dataloader, val_dataloader, model, optimizer, checkpoint_name=None, scheduler=None):
    patience = cfgs.get("patience", cfgs["max_epochs"])
    curr_patience = 0
    metric_names = cfgs.get("early_stopping", None)
    metric_modes = cfgs.get("early_stopping_mode", ['max'])
    
    # Get gradient accumulation steps (default to 1 if not specified)
    gradient_accumulation_steps = cfgs.get("gradient_accumulation_steps", 1)
    
    early_stopping = False
    if metric_names is not None:
        early_stopping = True
        best_metric_init = {}
        for metric_name, metric_mode in zip(metric_names, metric_modes):
            if metric_mode == "max":
                best_metric_init[metric_name] = -999
            else:
                best_metric_init[metric_name] = 999
    
    if cfgs["restore"]:
        chkpt_name = checkpoint_name if checkpoint_name is not None else 'checkpoint_last.pth.tar'
        chkpt_file = os.path.join(cfgs["experiment_dir"], chkpt_name)
        if os.path.exists(chkpt_file):
            chkpt = torch.load(chkpt_file)
            global_step = chkpt["step"]
            start_epoch = chkpt["epoch"]
            if (cfgs["model"] == "dsmnet") & (cfgs.get("include_autoencoder")==True):
                model.load_state_dict(chkpt["state_dict"], strict=False)
            else:
                model.load_state_dict(chkpt["state_dict"])
                optimizer.load_state_dict(chkpt["optimizer"])
                if scheduler:
                    scheduler.load_state_dict(chkpt["scheduler"])
            if cfgs["early_stopping"] is not None:
                curr_patience = chkpt.get("patience", 0)
                best_metric = chkpt.get("best_metric", best_metric_init)
        else:
            global_step = 0
            start_epoch = 0
            if cfgs["early_stopping"] is not None:
                best_metric = best_metric_init
                curr_patience = 0
    else:
        global_step = 0
        start_epoch = 0
        if cfgs["early_stopping"] is not None:
            best_metric = best_metric_init
            curr_patience = 0
    if curr_patience == patience:
        return

    for epoch in range(start_epoch, cfgs["max_epochs"]):
        model.train()
        loss_train = AverageMeter()
        # Reset gradient accumulation counter
        accumulation_counter = 0
        
        for _, image, gt in tqdm(train_dataloader, desc=f"Epoch {epoch+1}: training ..."):
            accumulation_counter += 1
            
            image = data_to_device(image, device=cfgs["device"])
            gt = data_to_device(gt, device=cfgs["device"])
            losses, pred = model(image, gt)
            loss_total = losses["loss_total"]
            
            # Scale the loss based on accumulation steps
            loss_total = loss_total / gradient_accumulation_steps
            loss_train.update(loss_total.item() * gradient_accumulation_steps, len(image))

            # Backward pass
            loss_total.backward()
            
            # Gradient accumulation - only update weights after specified number of steps
            if accumulation_counter % gradient_accumulation_steps == 0:
                global_step += 1
                log_flag = (global_step % cfgs["log_interval"] == 0)
                
                # Optimize
                optimizer.step()
                optimizer.zero_grad()
                
                if (cfgs.get("lr_policy", "constant") != "reduceonplateau") & (scheduler is not None):
                    scheduler.step()
                    
                if log_flag:
                    log_dict = {
                        'step': global_step,
                        'epoch': epoch
                    }
                    log_dict.update({'train/'+key: loss for key, loss in losses.items()})
                    if scheduler:
                        log_dict.update({'lr': float(optimizer.param_groups[0]['lr'])})
                    log_dict.update(model.vis(image, pred, gt))
                    logger.log(log_dict)
        
        logger.log({
            'epoch': epoch,
            'train/loss_avg': loss_train.avg
        })

        model.eval()
        loss_val = AverageMeter()
        eval_dict = UpdatableDict()
        skip_flag = False
        with torch.no_grad():
            for _, image, gt in tqdm(val_dataloader, desc=f"Epoch {epoch+1}: validating ..."):
                tic = time.perf_counter()

                image = data_to_device(image, device=cfgs["device"])
                gt = data_to_device(gt, device=cfgs["device"])
                losses, pred, eval_params = model(image, gt)
                if not cfgs.get("rcnn", False):
                    loss_val.update(losses["loss_total"].item(), len(image))
                eval_dict.update(eval_params)
                toc = time.perf_counter()
                if toc - tic > 10:
                    print(f"Epoch {epoch+1}: validation takes too long, skipping ...")
                    skip_flag = True
                    break
        save_dict = {
            'step': global_step,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if scheduler:
            save_dict.update({'scheduler': scheduler.state_dict()})
        log_dict = {'epoch': epoch}
        if not cfgs.get("rcnn", False):
            log_dict.update({'val/loss_total': loss_val.avg})
        log_dict.update(model.vis(image, pred, gt, True))
        if (not skip_flag) & early_stopping:
            eval_res = model.evaluate(eval_dict())
            log_dict.update(eval_res)
            no_stop_flag = []
            for metric_name, metric_mode in zip(metric_names, metric_modes):
                if ((metric_mode == 'max') & (log_dict[metric_name] > best_metric[metric_name])) | \
                ((metric_mode == 'min') & (log_dict[metric_name] < best_metric[metric_name])):
                    best_metric[metric_name] = log_dict[metric_name]
                    no_stop_flag.append(True)
                    curr_patience = 0
                else:
                    no_stop_flag.append(False)
            
            if (cfgs.get("lr_policy", "constant") == "reduceonplateau") & (scheduler is not None):
                scheduler.step(eval_res["rmse"])

            if not any(no_stop_flag):
                curr_patience += 1
            else:
                save_dict.update({
                    'best_metric': best_metric,
                })
                for metric_name, no_stop in zip(metric_names, no_stop_flag):
                    if no_stop:
                        if '/' in metric_name:
                            metric_name = metric_name.split('/')[-1]
                        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_best_{:s}.pth.tar'.format(metric_name)))
        save_dict.update({
            "patience": curr_patience
        })
        logger.log(log_dict)
        
        if (epoch % cfgs["checkpoint_interval"]) == 0:
            torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_{:03d}.pth.tar'.format(epoch))) 
        
        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], 'checkpoint_last.pth.tar'))

        if (not skip_flag) & early_stopping & (curr_patience == patience):
            print(f"Epoch {epoch+1}: maximum patience reached, early stopping ...")
            break

def main():
    args, unknown = parse_arguments()
    cfgs = {}

    if not args.restore:
        assert (args.config is not None) \
            & (args.exp_config is not None) \
            & os.path.isfile(args.config) \
            & os.path.isfile(args.exp_config), "Config files should be specified and exist"

        cfgs = load_yaml(args.config)
        cfgs.update(load_yaml(args.exp_config))
        cfgs["restore"] = False
        cfgs["overfit"] = args.overfit
        cfgs["checkpoint_dir"] = os.path.join(cfgs["checkpoint_dir"], cfgs["model"])
        cfgs["experiment_dir"] = os.path.join(cfgs["checkpoint_dir"], datetime.now().strftime('%y%m%d_%H%M%S'))

        if cfgs["overfit"]:
            cfgs["log_interval"] = 1
            #cfgs["early_stopping"] = None
            cfgs["checkpoint_interval"] = cfgs["max_epochs"]
            cfgs["batch_size"] = 1#2 if cfgs["model"] != "adabins_global" else 1
            cfgs["patience"] = cfgs["max_epochs"]
        
        print(f"Starting from {args.exp_config}...")

    else:
        assert (args.exp_config is not None) \
            & os.path.isfile(args.exp_config), "Experiment config file should be specified and exist"
        cfgs = load_yaml(args.exp_config)

        print(f"Restoring from {args.exp_config}...")
        cfgs['restore'] = True

    if unknown:
        assert (len(unknown)%2==0), "Misc variables should be in pairs, key and value"
        for key, value in zip(unknown[0::2], unknown[1::2]):
            cfgs[key] = convert_from_string(value)
    
    # Setup CUDA and handle multi-GPU
    cfgs["device"], gpu_list = setup_cuda(args.gpu_ids)
    cfgs["gpu_ids"] = gpu_list
    
    project = cfgs.get("project", 'GBH')
    runname = cfgs.get("name", None)
    
    # Create experiment directory
    os.makedirs(cfgs["experiment_dir"], exist_ok=True)
    
    # Use wandb if enabled, otherwise use dummy logger
    if args.no_wandb:
        logger = DummyLogger()
        cfgs["wandb_run_id"] = "dummy_run"
    else:
        # Use your wandb entity or the one from config
        entity = cfgs.get("wandb_entity", "ahmad-naghavi")
        try:
            logger = wandb.init(project=project, entity=entity, id=cfgs.get("wandb_run_id"), resume='must') if cfgs["restore"] else wandb.init(project=project, entity=entity, name=runname)
            cfgs["wandb_run_id"] = cfgs["wandb_run_id"] if cfgs["restore"] else logger.id
            logger.config.update(cfgs, allow_val_change=True)
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Falling back to dummy logger")
            logger = DummyLogger()
            cfgs["wandb_run_id"] = "dummy_run"

    print(cfgs)
    save_config(cfgs, os.path.join(cfgs["experiment_dir"], 'config.yaml'))
    
    seed = cfgs.get("seed", 42)
    fix_seed_for_reproducability(seed)

    train_loader, val_loader = get_train_val_dataloaders(cfgs)
    model, optimizer = get_model_and_optimizer(cfgs)
    model.to(cfgs["device"])
    
    # Handle multi-GPU - only use DataParallel if CUDA is available and multiple GPUs are specified
    if torch.cuda.is_available() and len(cfgs["gpu_ids"]) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfgs["gpu_ids"])
    
    # Watch model if using wandb
    if not args.no_wandb and isinstance(logger, wandb.sdk.wandb_run.Run):
        logger.watch(model)
    
    train(cfgs, logger, train_loader, val_loader, model, optimizer)


if __name__ == "__main__":
    main()

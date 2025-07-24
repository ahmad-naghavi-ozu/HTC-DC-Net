import torch
import argparse
import os
from tqdm import tqdm
from glob import glob
import wandb
from utils import AverageMeter, UpdatableDict, convert_from_string, data_to_device, load_yaml, \
    fix_seed_for_reproducability, save_config
from build import get_model_and_optimizer
from dataloaders import get_test_dataloaders
from cuda_setup import setup_cuda

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test configuration')
    parser.add_argument('--config', default=None, help='Specify a config file path')
    parser.add_argument('--vis', action='store_true', help='Visualize test results')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--gpu_ids', type=str, default="0", help='GPU IDs to use (comma-separated, e.g., "0,1,2")')
    parser.add_argument('misc', nargs='*', metavar='misc', help='Other variables')
    args = parser.parse_args()
    return args

class DummyLogger:
    """A dummy logger to use when wandb is disabled"""
    def __init__(self):
        print("Using dummy logger (wandb disabled)")
        
    def log(self, *args, **kwargs):
        pass
        
    def __setattr__(self, name, value):
        pass

def test(cfgs, test_dataloaders, model, logger):
    model.eval()
    chkpt_file = cfgs.get("test_checkpoint_file", 'checkpoint_best*.pth.tar')
    chkpt_files = glob(os.path.join(cfgs["experiment_dir"], chkpt_file))
    if len(chkpt_files) == 0:
        chkpt_files = [os.path.join(cfgs["experiment_dir"], 'checkpoint_last.pth.tar')]
    for chkpt_file in chkpt_files:
        filename = os.path.basename(chkpt_file)
        if cfgs["vis_test"]:
            vis_dir = os.path.join(cfgs["experiment_dir"], filename.split('.')[0].replace('checkpoint', 'result'))
            os.makedirs(vis_dir, exist_ok=True)
        else:
            vis_dir = None
        chkpt = torch.load(chkpt_file, map_location=cfgs["device"])
        epoch = chkpt["epoch"]
        
        # Handle loading state dict for DataParallel models
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(chkpt["state_dict"])
        else:
            model.load_state_dict(chkpt["state_dict"])

        save_dict = {'epoch': epoch}
        with torch.no_grad():
            for data_name, test_dataloader in test_dataloaders.items():
                loss_test = AverageMeter()
                eval_dict = UpdatableDict()
                for image_idx, image, gt in tqdm(test_dataloader, desc=f"Epoch {epoch+1}, {data_name}: testing ..."):
                    image = data_to_device(image, device=cfgs["device"])
                    gt = data_to_device(gt, device=cfgs["device"])

                    losses, pred, eval_params = model(image, gt)
                    loss_test.update(losses["loss_total"].item(), len(image))
                    eval_dict.update(eval_params)
                    # Handle DataParallel for visualization
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.vis(image, pred, gt, image_idx=image_idx, save=vis_dir)
                    else:
                        model.vis(image, pred, gt, image_idx=image_idx, save=vis_dir)
                
                # Get evaluation results
                if isinstance(model, torch.nn.DataParallel):
                    eval_res = model.module.evaluate(eval_dict())
                else:
                    eval_res = model.evaluate(eval_dict())
                    
                save_dict.update({
                    data_name: {
                        ** eval_res, 
                        'loss_total':loss_test.avg
                    }})
        save_dict = data_to_device(save_dict, 'cpu_test')
        if hasattr(logger, 'summary'):
            logger.summary['test'] = save_dict
        torch.save(save_dict, os.path.join(cfgs["experiment_dir"], filename.replace('checkpoint', 'result')))


def main():
    args = parse_arguments()
    cfgs = {}
    assert (args.config is not None) \
        & os.path.isfile(args.config), "Config file should be specified and exist"
    cfgs = load_yaml(args.config)
    cfgs["vis_test"] = args.vis
    cfgs["test"] = True
    if args.misc:
        assert (len(args.misc)%2==0), "Misc variables should be in pairs, key and value"
        for key, value in zip(args.misc[0::2], args.misc[1::2]):
            cfgs[key] = convert_from_string(value)
    if "exp_config" in cfgs:
        cfgs.update(load_yaml(cfgs["exp_config"]))
    print(f"Starting test from {args.config}...")
    print(cfgs)
    
    seed = cfgs.get("seed", 42)
    fix_seed_for_reproducability(seed)

    test_loaders = get_test_dataloaders(cfgs)
    
    # Setup CUDA and handle multi-GPU
    cfgs["device"], is_multi_gpu = setup_cuda(args.gpu_ids)
    cfgs["gpu_ids"] = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")] if is_multi_gpu else []
    
    if cfgs["model"] == "dsmnet":
        assert cfgs["include_autoencoder"] & cfgs["restore"], "DSMNet should be fully trained before test"
 
    project = cfgs.get("project", 'DFC2023S')
    # Use your wandb entity or the one from config
    entity = cfgs.get("wandb_entity", "ahmad-naghavi-ozu")
    
    if args.no_wandb:
        logger = DummyLogger()
    else:
        try:
            if "tade" in cfgs["model"]:
                save_config(cfgs, os.path.join(cfgs["experiment_dir"], "config_tade.yaml"))
                model, optimizer = get_model_and_optimizer(cfgs)
                logger = wandb.init(project=project, entity=entity)
                logger.config.update(cfgs, allow_val_change=True)
            else:
                model = get_model_and_optimizer(cfgs, True)
                logger = wandb.init(project=project, entity=entity, id=cfgs.get("wandb_run_id"), resume='must')
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Falling back to dummy logger")
            logger = DummyLogger()
    
    model.to(cfgs["device"])
    
    # Handle multi-GPU setup
    if is_multi_gpu:
        print(f"Using DataParallel with GPUs: {cfgs['gpu_ids']}")
        model = torch.nn.DataParallel(model, device_ids=cfgs["gpu_ids"])
    
    test(cfgs, test_loaders, model, logger)


if __name__ == "__main__":
    main()

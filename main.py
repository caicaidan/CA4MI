import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from ca4mi import CA4MI
from models import architure as network
import utils


# --------------------------------------------------- Main Function ----------------------------------------------------
def main(args):
    utils.create_checkpoint_structure(args)
    utils.copy_project_files_to_destination(args)

    print("=" * 100)
    print("Arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"\t{arg_name}: {arg_value}")
    print("=" * 100)

    ACC, BWT = [], []

    for run_index in range(args.n_runs):
        args.seed = run_index
        set_seed(args.seed)
        print(f"Run {run_index + 1}/{args.n_runs}")

        # Initialize data loader and model
        dataloader = initialize_dataloader(args)
        net = network.Cls_HEAD(args).to(args.device)
        ca4mi = CA4MI(net, args, network=network)

        # Initialize result matrices
        acc = np.zeros((args.n_subjects, args.n_subjects), dtype=np.float32)
        lss = np.zeros((args.n_subjects, args.n_subjects), dtype=np.float32)

        # Training and evaluation loop
        max_prototypes = args.max_prototypes
        source_center = []
        for t in range(args.n_subjects):
            print("=" * 250)
            dataset = dataloader.load_data(t)
            print(f"{' ' * 105} Dataset {t + 1:2d} ({dataset[t]['name']})")
            print("=" * 250)

            if args.use_prototypes == 'yes':
                ca4mi.Adversarial_training(t, dataset[t], prototypes=source_center)
                prototypes, _ = ca4mi.get_prototype_samples(dataset[t]['train'], ca4mi.load_current_models(t), t)
                source_center.append(prototypes)

                # reservoir sampling for prototypes management
                total_prototypes = sum([proto.size(0) for proto in source_center])
                if total_prototypes > max_prototypes:
                    print(
                        f"Total number of prototypes {total_prototypes} exceeds maximum limit {max_prototypes}, performing reservoir sampling")
                    source_center = ca4mi.reservoir_sampling(source_center, max_prototypes)
            else:
                ca4mi.Adversarial_training(t, dataset[t], prototypes=None)

            # Evaluate on all previous subjects
            evaluate_previous_subjects(ca4mi, dataset, t, acc, lss)

        # Save accuracy results
        output_path = os.path.join(args.checkpoint, f"{args.experiment}_{args.n_subjects}_subjects_seed_{args.seed}.txt")
        print(f"\nSaved accuracies at {output_path}")
        np.savetxt(output_path, acc, '%.6f')

        # Log results
        avg_acc, gem_bwt = utils.print_log_acc_bwt(acc, lss, output_path=args.checkpoint, run_id=run_index)
        ACC.append(avg_acc)
        BWT.append(gem_bwt)

    display_final_results(ACC, BWT, args.n_runs)

# ------------------------------------------------ Helper Functions ----------------------------------------------------

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_dataloader(args):
    """Initialize data loader based on experiment type."""
    if args.experiment == 'openBMI':
        from dataloaders import openBMI as factory
    elif args.experiment == 'bci-competition-IV2a':
        from dataloaders import bcicomp2a as factory
    elif args.experiment == 'bci-competition-IV2b':
        from dataloaders import bcicomp2b as factory
    else:
        raise NotImplementedError("Experiment type not recognized")

    dataloader = factory.IncrementalDataStreaming(args)
    args.subject_class = dataloader.sub_cls
    return dataloader

def evaluate_previous_subjects(ca4mi, dataset, t, acc, lss):
    """Evaluate the model on all previously seen subjects."""
    for u in range(t + 1):
        test_result = ca4mi.test_previous_subjects(dataset[u]['test'], u, model=ca4mi.load_previous_model(u))
        print(f">>> Test on {dataset[u]['name']:15s}: loss={test_result['loss_t']:.3f}, "
              f"acc={test_result['acc_t']:5.1f}% <<<")
        acc[t, u] = test_result['acc_t']
        lss[t, u] = test_result['loss_t']

def display_final_results(ACC, BWT, n_runs):
    """Display the average results over multiple runs."""
    mean_accuracy, std_accuracy = np.mean(ACC), np.std(ACC)
    mean_forgetting, std_forgetting = np.mean(BWT), np.std(BWT)

    print("*" * 100)
    print(f"Average over {n_runs} runs")
    print(f"ACC.AVG: {mean_accuracy:.4f}% ± {std_accuracy:.4f}")
    print(f"BWT.AVG: {mean_forgetting:.4f}% ± {std_forgetting:.4f}")
    print("All Done!!")

# --------------------------------------------------- Run Script ----------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Continual Adaptation for Motor Imagery EEG Decoding (CA4MI)")
    parser.add_argument('--config', type=str, default='./configs/bcicomp2a.yml')
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)
    main(args)

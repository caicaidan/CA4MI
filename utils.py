import os
import numpy as np
import pickle
import time
import uuid
from subprocess import call


def copy_project_files_to_destination(args):
    current_working_directory = os.getcwd()
    destination_diretory = os.path.join(args.checkpoint, 'code') + '/'
    if not os.path.exists(destination_diretory):
        os.mkdir(destination_diretory)

    def get_full_folder_path(folder_name):
        return os.path.join(current_working_directory, folder_name)

    folders = [get_full_folder_path(item) for item in
               ['dataloaders', 'models', 'configs', 'main.py', 'sil_trans.py', 'utils.py']]
    for folder in folders:
        call('cp -rf {} {}'.format(folder, destination_diretory), shell=True)


def create_checkpoint_structure(args):
    uid = uuid.uuid4().hex
    if args.checkpoint is None:
        os.mkdir('checkpoints')
        args.checkpoint = os.path.join('./checkpoints/', uid)
        os.mkdir(args.checkpoint)
    else:
        if not os.path.exists(args.checkpoint):
            os.mkdir(args.checkpoint)
        args.checkpoint = os.path.join(args.checkpoint, uid)
        os.mkdir(args.checkpoint)

#
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d} | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


def report_val(res):
    # Validation performance
    print(
        ' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
            res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def print_log_acc_bwt(acc, lss, output_path, run_id):
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i, j]), end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0] - 1, :])
    print('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    gem_bwt = sum(acc[-1] - np.diag(acc)) / (len(acc[-1]) - 1)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print('BWT: {:5.2f}%'.format(gem_bwt))
    print('*' * 100)
    print('Done!')

    logs = {}
    logs['name'] = output_path
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print("Log file saved in ", path)
    return avg_acc, gem_bwt

def loader_size(data_loader):
    return data_loader.dataset.__len__()


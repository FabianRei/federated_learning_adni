import os, sys, inspect
# this is done to run things from console
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dl.neural_network.train_h5 import train_h5
import GPUtil
import multiprocessing as mp
import time
import datetime


def job_generator(jobs):
    for job in jobs:
        yield job


def run_jobs(jobs):
    device_ids = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    print(device_ids)
    job_gen = job_generator(jobs)
    process_dict = {}
    while True:
        try:
            if process_dict == {}:
                for device in device_ids:
                    job = next(job_gen)
                    print(f"Running {job} on GPU {device}")
                    sub_proc = mp.Process(target=train_h5, args=[job[0]], kwargs={'gpu_device': device, **job[1]})
                    process_dict[str(device)] = sub_proc
                    sub_proc.start()
            for device, proc in process_dict.items():
                if not proc.is_alive():
                    job = next(job_gen)
                    print(f"Running {job} on GPU {device}")
                    sub_proc = mp.Process(target=train_h5, args=[job[0]], kwargs={'gpu_device': device, **job[1]})
                    process_dict[str(device)] = sub_proc
                    sub_proc.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in process_dict.values():
        proc.join()


if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

r'''
###############################################
##############Past runs:#######################
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20,
             'lr': 0.001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'],
             'binning': 20, 'lr': 0.001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33,
             'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'lr': 0.01},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'lr': 0.01}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time() - full_start))} hours:min:seconds")
#############################################################################3
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.00001}]
    

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/more_one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': '', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001},
            {'extra_info': '', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'regression': True, 'lr': 0.0001}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': 'pretrained_20bins', 'pretrained': True, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_20bins', 'pretrained': False, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'pretrained_50epochs_20bins', 'pretrained': True, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_50epochs_20bins', 'pretrained': False, 'num_epochs': 50, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'pretrained_0_33_decrease_20bins', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20},
            {'extra_info': 'non_pretrained_0_33_decrease_20bins', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33, 'label_names': ['label_suvr', 'label_amyloid'], 'binning': 20}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###############################################
if __name__ == '__main__':
    full_start = time.time()
    h5_file = '/scratch/reith/fl/experiments/one_slice_dataset/slice_data.h5'
    jobs = [{'extra_info': 'pretrained', 'pretrained': True},
            {'extra_info': 'non_pretrained', 'pretrained': False},
            {'extra_info': 'pretrained_50epochs', 'pretrained': True, 'num_epochs': 50},
            {'extra_info': 'non_pretrained_50epochs', 'pretrained': False, 'num_epochs': 50},
            {'extra_info': 'pretrained_0_33_decrease', 'pretrained': True, 'decrease_after': 3, 'rate_of_decrease': 0.33},
            {'extra_info': 'non_pretrained_0_33_decrease', 'pretrained': False, 'decrease_after': 3, 'rate_of_decrease': 0.33}]

    jobs = [(h5_file, job) for job in jobs]
    run_jobs(jobs)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
'''
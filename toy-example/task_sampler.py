import torch
import math
import random
import numpy as np

def task_shift(task):
    """
    Fetch shift amount for task.
    """
    return math.pi * task /12.0


def f(X, shift=0):
    """
    Torch-compatible objective function for the target_task
    """
    f_X =  torch.sin(X/2 + math.pi/2 + shift)
    return f_X

def get_batch(size=10, num_tasks=5):
    
    ix = random.sample(np.arange(-10, 10, 1).tolist(), size*2)
    task = np.random.randint(10,10+num_tasks)
    X_spt, X_qry = torch.Tensor(ix[:size]).reshape(-1,1), torch.Tensor(ix[size:]).reshape(-1,1)
    Y_spt, Y_qry = f(X_spt, shift=task_shift(task)), f(X_qry, shift=task_shift(task))
    
    return X_spt/10, Y_spt, X_qry/10, Y_qry

def get_batch_test(spt_size=4, task = 17):

    X_qry = torch.Tensor(np.arange(-10, 10, 0.1)).reshape(-1,1)
    ix = random.sample(np.arange(-10, 10, 0.1).tolist(), spt_size)
    
    X_spt = torch.Tensor(ix).reshape(-1,1)
    y_spt = f(X_spt, shift=task_shift(task))
    y_qry = f(X_qry, shift=task_shift(task))
    
    return X_spt/10, y_spt, X_qry/10, y_qry

def get_batch_val(spt_size=5, task = 16):

    #X_qry = torch.Tensor(np.arange(-10, 10, 0.1)).reshape(-1,1)
    ix1 = random.sample(np.arange(-10, 10, 1).tolist(), spt_size)
    ix2 = random.sample(np.arange(-10, 10, 1).tolist(), spt_size)
    
    X_spt = torch.Tensor(ix1).reshape(-1,1)
    X_qry = torch.Tensor(ix2).reshape(-1,1)

    y_spt = f(X_spt, shift=task_shift(task))
    y_qry = f(X_qry, shift=task_shift(task))
    
    return X_spt/10, y_spt, X_qry/10, y_qry
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import functools
import os
import subprocess
import tensorflow as tf
import herring.tensorflow as herring

def init_dist():
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[herring.local_rank()], 'GPU')

def get_dist_info():
    return herring.rank(), herring.local_rank(), herring.size(), herring.local_size() #TODO return a dict instead

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _, _, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def broadcast_weights(runner):
    print('Rank {} broadcasting'.format(runner.rank))
    herring.broadcast_variables(runner.model.variables + runner.optimizer.variables(), root_rank=0)
    # herring.broadcast_variables(runner.optimizer.variables(), root_rank=0)
    print('Variable broadcast done.')

def get_distributed_tape(tape):
    return herring.DistributedGradientTape(tape)

def get_barrier():
    return herring.oob_allreduce(tf.constant(0, dtype=tf.float32))

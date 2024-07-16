#!/bin/env python
import logging
import time

import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from concurrent.futures import ProcessPoolExecutor, TimeoutError

def run_xtb(xtb_job, logging_queue):
    ''' subprocess for running xtb in parallel '''
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

    if xtb_job.calculation_terminated_normally():
        print(f"XTB job {xtb_job.jobname.split()[-1]} has been finished, skip this job...")
        logger.info(f"XTB job {xtb_job.jobname.split()[-1]} has been finished, skip this job...")
    else:
        print(f"running XTB job {xtb_job.jobname.split()[-1]} on PID {mp.current_process().pid}")
        logger.info(f"running XTB job {xtb_job.jobname.split()[-1]} on PID {mp.current_process().pid}")
        xtb_job.execute()

def run_crest(crest_job, logging_queue):
    ''' subprocess for running crest in parallel '''
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

    if crest_job.calculation_terminated_normally():
        print(f"CREST job {crest_job.jobname} has been finished, skip this job...")
        logger.info(f"CREST job {crest_job.jobname} has been finished, skip this job...")
    else:
        print(f"running CREST job {crest_job.jobname} on PID {mp.current_process().pid}")
        logger.info(f"running CREST job {crest_job.jobname} on PID {mp.current_process().pid}")
        result = crest_job.execute()
        if result.returncode == 0:
            print(f"CREST job {crest_job.jobname} is finished.")
            logger.info(f"CREST job {crest_job.jobname} is finished.")
        else:
            print(f"Command failed for CREST job {crest_job.jobname} with the following error message:")
            logger.info(f"Command failed for CREST job {crest_job.jobname}, check job log file for detailed information")
            print(result.stderr)

def run_gsm(gsm_job, logging_queue):
    ''' subprocess for running gsm in parallel '''
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

    if gsm_job.calculation_terminated_normally():
        print(f"GSM job {gsm_job.jobname} has been finished, skip this job...")
        logger.info(f"GSM job {gsm_job.jobname} has been finished, skip this job...")
    #Zhao's note: output file exists, but calculation ends with error#
    elif gsm_job.output_file_exist() and not gsm_job.calculation_terminated_without_error():
        print(f"GSM job {gsm_job.jobname} has been finished but has error, skip this job...")
        logger.info(f"GSM job {gsm_job.jobname} has been finished but has error, skip this job...")
    else:
        print(f"running GSM job {gsm_job.jobname} on PID {mp.current_process().pid}")
        logger.info(f"running GSM job {gsm_job.jobname} on PID {mp.current_process().pid}")
        start = time.time()
        result= gsm_job.execute()
        end   = time.time()
        if result.returncode == 0:
            print(f"GSM job {gsm_job.jobname} is finished, with running time {end-start:.1f}s")
            logger.info(f"GSM job {gsm_job.jobname} is finished, with running time {end-start:.1f}s")
        else:
            print(f"Command failed for GSM job {gsm_job.jobname} with the following error message:")
            logger.info(f"Command failed for GSM job {gsm_job.jobname}, check job log file for detailed information")
            print(result.stderr)

def run_pysis(pysis_job, logging_queue, timeout=3600):
    ''' subprocess for running pysis in parallel '''
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

    if pysis_job.calculation_terminated_normally():
        print(f"PYSIS job {pysis_job.jobname} has been finished, skip this job...")
        logger.info(f"PYSIS job {pysis_job.jobname} has been finished, skip this job...")
    else:
        print(f"running PYSIS job {pysis_job.jobname} on PID {mp.current_process().pid}")
        logger.info(f"running PYSIS job {pysis_job.jobname} on PID {mp.current_process().pid}")
        result = pysis_job.execute(timeout=timeout)
        if result.returncode == 0:
            print(f"PYSIS job {pysis_job.jobname} is finished.")
            logger.info(f"PYSIS job {pysis_job.jobname} is finished.")
        else:
            print(f"Command failed for PYSIS job {pysis_job.jobname} with the following error message:")
            logger.info(f"Command failed for PYSIS job {pysis_job.jobname}, check job log file for detailed information")
            print(result.stderr)
def run_ssm(ssm_job):
    ''' subprocess for running ssm in parallel '''
    if ssm_job.calculation_terminated_normally():
        print(f"SSM job {ssm_job.jobname} has been finished, skip this job...")
    else:
        print(f"running SSM job {ssm_job.jobname} on PID {mp.current_process().pid}")
        start = time.time()
        result= ssm_job.execute()
        end   = time.time()
        if result.returncode == 0:
            print(f"SSM job {ssm_job.jobname} is finished, with running time {end-start:.1f}s")
        else:
            print(f"Command failed for SSM job {ssm_job.jobname} with the following error message:")
            print(result.stderr)

def run_gauxtb(gau_job):
    ''' subprocess for running gau-xtb in parallel '''

    if gau_job.job_finished():
        print(f"Gaussian job {gau_job.jobname} has been finished, skip this job...")
    else:
        print(f"running Gaussian job {gau_job.jobname} on PID {mp.current_process().pid}")
        start = time.time()
        result= gau_job.execute()
        end   = time.time()
        if result.returncode == 0:
            print(f"Gaussian job {gau_job.jobname} is finished, with running time {end-start:.1f}s")
        else:
            print(f"Command failed for Gaussian job {gau_job.jobname} with the following error message:")
            print(result.stderr)

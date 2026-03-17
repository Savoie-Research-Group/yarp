import subprocess
import psutil

class BaseJobManager:
    def submit(self, script_path: str) -> str: raise NotImplementedError
    def is_running(self, job_id: str) -> bool: raise NotImplementedError

class LocalJobManager(BaseJobManager):
    def submit(self, script_path: str) -> str:
        # Runs the bash script locally and detaches it
        process = subprocess.Popen(["bash", str(script_path)])
        return str(process.pid)

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        return psutil.pid_exists(int(job_id))

class SlurmJobManager(BaseJobManager):
    def submit(self, script_path: str) -> str:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        try:
            result = subprocess.run(["squeue", "-j", str(job_id), "-h"], capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

class QSEJobManager(BaseJobManager):
    def submit(self, script_path: str) -> str:
        # Assuming qsub returns the job ID directly
        result = subprocess.run(["qsub", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        try:
            # qstat returns exit code 0 if job is in queue/running, non-zero if finished/not found
            result = subprocess.run(["qstat", "-j", str(job_id)], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

def get_job_manager(scheduler_type: str) -> BaseJobManager:
    if scheduler_type.lower() == "slurm":
        return SlurmJobManager()
    elif scheduler_type.lower() == "qse":
        return QSEJobManager()
    elif scheduler_type.lower() == "local":
            return LocalJobManager()
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
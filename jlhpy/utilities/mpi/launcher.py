from mpi4py import MPI
import logging

def launch():
    """Spawns MPI processes to run parallel function."""
    logger = logging.getLogger(__name__)

    version = MPI.Get_version()
    logger.info("MPI version: %s" % str(version))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger.info("Current MPI size is: %d" % size)
    universe_size = comm.Get_attr(MPI.UNIVERSE_SIZE)
    logger.info("MPI universe size is: %d" % universe_size)

    soft = MPI.INFO_ENV.get("soft")
    logger.info("MPI.INFO_ENV 'soft' maximum number of processors is: %s" % soft)

    maxprocs = MPI.INFO_ENV.get("maxprocs")
    logger.info("MPI.INFO_ENV 'maxprocs' maximum number of processes is: %s" % maxprocs)

#
# See top-level LICENSE.rst file for Copyright information
#
"""
desispec.workflow.schedule
==========================

Tools for scheduling MPI jobs using mpi4py
"""
from mpi4py import MPI
import numpy as np
from logging import getLogger

class Schedule:

    def __init__(self, workfunc, **kwargs):
        """
        Intialize class for scheduling MPI jobs using mpi4py 
        
        Args:
            workfunc: function to do each MPI job defined using 
                      def workfunc(comm,job):
                          where comm is an MPI communicator 
                          and job is the index of the job 
            
        Keyword Args:
            comm:       MPI communicator (default=None)
            njobs:      number of jobs (default=2)
            group_size: number of MPI processes per job (default=1) 
        """
        # user provided function that will do the work
        self._workfunc = workfunc

        self.comm       = kwargs.get('comm',None)
        self.njobs      = kwargs.get('njobs',2)
        self.group_size = kwargs.get('group_size',1)

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.log = getLogger(__name__)
        
        # numpy array for sending and receiving job indices
        self.job_buff = np.zeros(1,dtype=np.int32)

        if self.group_size > self.size - 1:
            raise Exception("can't have group_size larger than world size - 1")

        self.Ngroups  = (self.size-1) // self.group_size
        self.group    = (self.rank-1) // self.group_size

        # assign rank=0 and 'extra' ranks to group Ngroups
        # only ranks with group < Ngroups participate as workers
        if self.rank > self.group_size * self.Ngroups or self.rank == 0:
            self.group = self.Ngroups

        self.groupcomm = self.comm.Split(color=self.group)
        self.grouprank = self.groupcomm.Get_rank()

        if self.group_size != self.groupcomm.Get_size() and self.rank != 0:
            print(self.rank,self.group_size,self.groupcomm.Get_size(),flush=True)
            raise Exception("can't have group_size != group_size")
        
    def _assign_job(self,worker,job):

        """
        Assign job to a group of processes 
        
        Args:
            worker: index of group of processes 
            job:    index of job to be assigned
            
        Returns:
            reqs: list of mpi4py.MPI.Request objects, one for each process in group 
        """

        # assign job to all processes in group worker
        # and return list of handles corresponding to
        # confirmation of completion of current job
        reqs = []
        for grouprank in range(self.group_size):
            destrank = worker * self.group_size + grouprank + 1
            self.job_buff[0] = job
            self.comm.Send(self.job_buff,dest=destrank)
            reqs.append(self.comm.Irecv(self.job_buff,source=destrank))
        return reqs

    def _checkreqlist(self,reqs):

        """
        Check for completion of jobs by all processes in group 
        
        Args:
            reqs: list of mpi4py.MPI.Request objects, one for each process in group 
            
        Returns:
            bool: True if all messages corresponding to reqa received, False otherwise 
        """
        
        # check if all processes with handles in reqs have reported back
        for req in reqs:
            if not req.Test():
                return False
        return True

    def _schedule(self):

        """
        Schedule, run and assign processes for all jobs in this object        
        """
        # bookkeeping
        waitlist        = [] # message handles for pending worker groups
        worker_groups   = [] # worker assigned

        # start by assigning Ngroups jobs to each of the Ngroup groups
        nextjob=0
        for job in range(self.Ngroups):
            worker = nextjob
            reqs=self._assign_job(worker,nextjob)
            waitlist.append(reqs)
            worker_groups.append(worker)
            nextjob += 1

        # the scheduler waits for jobs to be completed;
        # when one is complete it assigns the next job
        # until there are none left
        Ncomplete = 0
        while(Ncomplete < self.njobs):
            # iterate over list of currently pending group of processes
            for i in range(len(waitlist)):
                # check for completion of all processes in this worker group
                if self._checkreqlist(waitlist[i]):
                    # all ranks group doing job corresponding to place i in waitlist
                    # have returned; identify this worker group and remove it from the
                    # waitlist and worker list
                    worker = worker_groups[i]
                    Ncomplete += 1
                    waitlist.pop(i)
                    worker_groups.pop(i)
                    if nextjob < self.njobs:
                        # more jobs to do; assign processes in group worker
                        # the job with index nextjob, increment
                        reqs=self._assign_job(worker,nextjob)
                        waitlist.append(reqs)
                        worker_groups.append(worker)
                        nextjob += 1
                    # waitlist has been modified so exit waitlist loop with break
                    break

        # no more jobs to assign; dismiss all processes in all groups by 
        # assigning job = -1, causing all workers processes to return
        for worker in range(self.Ngroups): 
            self._assign_job(worker,-1)

        return 
    
    def _work(self):
        """
        Listen for job assignments and run workfunc         
        """
        # listen for job assignments from the scheduler
        while True:
            self.comm.Recv(self.job_buff,source=0) # receive assignment from rank=0 scheduler
            job = self.job_buff[0]                 # unpack job index
            if job < 0: return                     # job < 0 means no more jobs to do
            try:
                self._workfunc(self.groupcomm,job) # call work function for job
            except Exception as e:
                self.log.error(f'FAILED: call to workfunc')
                self.log.error(e)
            self.comm.Isend(self.job_buff,dest=0)  # send non-blocking message on completion
            
        return

    def run(self):
        """
        Run schedulers and workers for this object          
        """
        # main function of class
        if self.rank==0:
            self._schedule() # run scheduler on rank = 0
        elif self.group < self.Ngroups:
            self._work()     # run worker on all other ranks

        self.comm.barrier()

        return

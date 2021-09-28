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

    def __init__(self, workfunc, comm=None, njobs=2, group_size=1):
        """
        Intialize class for scheduling MPI jobs using mpi4py 
        
        Args:
            workfunc: function to do each MPI job defined using 
                      def workfunc(groupcomm,job):
                          where groupcomm is an MPI communicator 
                          and job is an integer in the range 0 to njobs - 1
                      
            
        Keyword Args:
            comm:       MPI communicator (default=None)
            njobs:      number of jobs (default=2)
            group_size: number of MPI processes per job (default=1) 
        
        Initialization of this class results in 
            ngroups = (comm.Get_size() - 1) // group_size 
        new communicators (groups) being created, each with size group_size, 
        using comm.Split.
    
        Functionality of this class is provided via the Schedule.run method. The 
        process in comm with 
            rank = 0 
        will be dedicated to scheduling, while processes with 
            rank > ngroups * group_size 
        will remain idle, and processes with 
            0 < rank < ngroups * group_size 
        will run workfunc in groups of size group_size.  
        
        In the case  
            njobs >= ngroups 
        all ranks in each of the groups will first be assigned to call
        workfunc with arguments job = 0 to job = ngroups-1, in parallel. The 
        first group to finish workfunc will then call workfunc with job = ngroups, 
        the next group to finish will call workfunc with job = ngroups + 1, and so 
        on, until workfunc has returned for all njobs values of job. 
        
        In the case  
            njobs < ngroups 
        then only ranks assigned to the first njobs groups will run workfunc, 
        while the rest will remain idle, until workfunc has returned for all njobs 
        values of job.    
        
        """
        # user provided function that will do the work
        self._workfunc = workfunc

        self.comm       = comm 
        self.njobs      = njobs
        self.group_size = group_size

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.log = getLogger(__name__)
        
        # numpy array for sending and receiving job indices
        self.job_buff = np.zeros(1,dtype=np.int32)

        if self.group_size > self.size - 1:
            raise Exception("can't have group_size larger than world size - 1")

        # set number of groups and group for this rank 
        self.ngroups  = (self.size-1) // self.group_size
        self.group    = (self.rank-1) // self.group_size

        # assign rank=0 and 'extra' ranks to group ngroups
        # only ranks with group < ngroups participate as workers
        if self.rank > self.group_size * self.ngroups or self.rank == 0:
            self.group = self.ngroups

        # generate a new communicator for ngroups processes of size group_size 
        self.groupcomm = self.comm.Split(color=self.group)

        # check for consistency between specified group_size and that of new communicator 
        if self.group_size != self.groupcomm.Get_size() and self.rank != 0:
            self.log.error(f'FAILED: rank {self.rank} with group_size = '+
                           f'{self.group_size} and groupcomm.Get_size() returning '+
                           f'{self.groupcomm.Get_size()}')
            raise Exception("inconsistent group size")
        
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
            if job >= 0: reqs.append(self.comm.Irecv(self.job_buff,source=destrank))
        return reqs

    def _checkreqlist(self,reqs):

        """
        Check for completion of jobs by all processes in group 
        
        Args:
            reqs: list of mpi4py.MPI.Request objects, one for each process in group 
            
        Returns:
            bool: True if all messages corresponding to reqs received, False otherwise 
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

        # start by assigning ngroups jobs, one to each of the ngroups groups
        nextjob=0
        for job in range(self.ngroups):
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
        for worker in range(self.ngroups): 
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
                self.log.error(f'FAILED: call to workfunc for job {job}'+
                               f' on rank {self.rank}')
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
        elif self.group < self.ngroups:
            self._work()     # run worker on all other ranks

        self.comm.barrier()

        return

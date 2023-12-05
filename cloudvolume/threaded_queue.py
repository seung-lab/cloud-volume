from __future__ import print_function

from six.moves import queue as Queue
from six.moves import range
from functools import partial
import threading
import time

from tqdm import tqdm

DEFAULT_THREADS = 20

class ThreadedQueue(object):
  """Grant threaded task processing to any derived class."""
  def __init__(self, n_threads, queue_size=0, progress=None):
    self._n_threads = n_threads

    self._queue = Queue.Queue(maxsize=queue_size) # 0 = infinite size
    self._error_queue = Queue.Queue(maxsize=queue_size)
    self._threads = ()
    self._terminate = threading.Event()

    self._processed_lock = threading.Lock()
    self.processed = 0
    self._inserted = 0

    self.with_progress = progress

    self.start_threads(n_threads)

  @property
  def pending(self):
      return self._queue.qsize()

  def put(self, fn):
    """
    Enqueue a task function for processing.

    Requires:
      fn: a function object that takes one argument
        that is the interface associated with each
        thread.

        e.g. def download(api):
               results.append(api.download())

             self.put(download)

    Returns: self
    """
    self._inserted += 1
    self._queue.put(fn, block=True)
    return self

  def start_threads(self, n_threads):
    """
    Terminate existing threads and create a 
    new set if the thread number doesn't match
    the desired number.

    Required: 
      n_threads: (int) number of threads to spawn

    Returns: self
    """
    if n_threads == len(self._threads):
      return self
    
    # Terminate all previous tasks with the existing
    # event object, then create a new one for the next
    # generation of threads. The old object will hang
    # around in memory until the threads actually terminate
    # after another iteration.
    self._terminate.set()
    self._terminate = threading.Event()

    threads = []

    for _ in range(n_threads):
      worker = threading.Thread(
        target=self._consume_queue, 
        args=(self._terminate,)
      )
      worker.daemon = True
      worker.start()
      threads.append(worker)

    self._threads = tuple(threads)
    return self

  def are_threads_alive(self):
    """Returns: boolean indicating if any threads are alive"""
    return any(map(lambda t: t.is_alive(), self._threads))

  def kill_threads(self):
    """Kill all threads."""
    self._terminate.set()
    while self.are_threads_alive():
      time.sleep(0.001)
    self._threads = ()
    return self

  def _initialize_interface(self):
    """
    This is used to initialize the interfaces used in each thread.
    You should reimplement it in subclasses. For example, return
    an API object, file handle, or network connection. The functions
    you pass into the self._queue will get it as the first parameter.

    e.g. an implementation in a subclass.
 
        def _initialize_interface(self):
          return HTTPConnection()   

        def other_function(self):
          def threaded_file_read(connection):
              # do stuff

          self._queue.put(threaded_file_handle)

    Returns: Interface object used in threads
    """
    return None

  def _close_interface(self, interface):
    """Allows derived classes to clean up after a thread finishes."""
    pass

  def _consume_queue(self, terminate_evt):
    """
    This is the main thread function that consumes functions that are
    inside the _queue object. To use, execute self._queue(fn), where fn
    is a function that performs some kind of network IO or otherwise
    benefits from threading and is independent.

    terminate_evt is automatically passed in on thread creation and 
    is a common event for this generation of threads. The threads
    will terminate when the event is set and the queue burns down.

    Returns: void
    """
    interface = self._initialize_interface()

    while not terminate_evt.is_set():
      try:
        fn = self._queue.get(block=True, timeout=0.01)
      except Queue.Empty:
        continue # periodically check if the thread is supposed to die

      fn = partial(fn, interface)

      try:
        self._consume_queue_execution(fn)
      except Exception as err:
        self._error_queue.put(err)

    self._close_interface(interface)

  def _consume_queue_execution(self, fn):
    """
    The actual task execution in each thread. This
    is broken out so that exceptions can be caught
    in derived classes and allow them to manipulate 
    the errant task, e.g. putting it back in the queue
    for a retry.

    Every task processed will automatically be marked complete.

    Required:
      [0] fn: A curried function that includes the interface
              as its first argument.
    Returns: void
    """

    # `finally` fires after all success or exceptions
    # exceptions are handled in derived classes
    # and uncaught ones are caught as a last resort
    # in _consume_queue to be raised on the main thread.
    try:
      fn()
    finally:
      with self._processed_lock:
        self.processed += 1
        self._queue.task_done()

  def _check_errors(self):
    try:
      err = self._error_queue.get(block=False) 
      self._error_queue.task_done()
      self.kill_threads()
      raise err
    except Queue.Empty:
      pass

  def wait(self, progress=None):
    """
    Allow background threads to process until the
    task queue is empty. If there are no threads,
    in theory the queue should always be empty
    as processing happens immediately on the main thread.

    Optional:
      progress: (bool or str) show a tqdm progress bar optionally
        with a description if a string is provided
    
    Returns: self (for chaining)

    Raises: The first exception recieved from threads
    """
    if not len(self._threads):
      return self

    desc = None
    if type(progress) is str:
      desc = progress

    last = self._inserted
    with tqdm(total=self._inserted, disable=(not progress), desc=desc) as pbar:
      # Allow queue to consume, but check up on
      # progress and errors every tenth of a second
      while not self._queue.empty():
        size = self._queue.qsize()
        delta = last - size
        if delta != 0: # We should crash on negative numbers
          pbar.update(delta)
        last = size
        self._check_errors()
        time.sleep(0.015)

      # Wait until all tasks in the queue are 
      # fully processed. queue.task_done must be
      # called for each task.
      self._queue.join() 
      self._check_errors()

      final = self._inserted - last
      if final:
        pbar.update(final)

    if self._queue.empty():
      self._inserted = 0

    return self

  def __del__(self):
    self.wait() # if no threads were set the queue is always empty
    self.kill_threads()

  def __enter__(self):
    if self.__class__ is ThreadedQueue and self._n_threads == 0:
      raise ValueError("Using 0 threads in base class ThreadedQueue with statement will never exit.")

    self.start_threads(self._n_threads)
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    self.wait(progress=self.with_progress)
    self.kill_threads()

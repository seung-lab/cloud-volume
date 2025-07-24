import sys

from tqdm import tqdm

from .lib import totalfn
from .threaded_queue import ThreadedQueue, DEFAULT_THREADS

def schedule_threaded_jobs(
    fns, concurrency=DEFAULT_THREADS, 
    progress=None, total=None
  ):

  if total is None:
    try:
      total = len(fns)
    except TypeError: # generators don't have len
      pass

  if isinstance(progress, tqdm):
    pbar = progress
  else:
    pbar = tqdm(
      total=total, 
      desc=(progress if isinstance(progress, str) else None),
      disable=(not progress)
    )

  results = []
  
  def updatefn(fn):
    def realupdatefn(iface):
      res = fn()
      pbar.update(1)
      results.append(res)
    return realupdatefn

  with ThreadedQueue(n_threads=concurrency) as tq:
    for fn in fns:
      tq.put(updatefn(fn))

  pbar.close()
  return results

def schedule_green_jobs(
    fns, concurrency=DEFAULT_THREADS, 
    progress=None, total=None
  ):
  import gevent.pool

  if total is None:
    try:
      total = len(fns)
    except TypeError: # generators don't have len
      pass

  if isinstance(progress, tqdm):
    pbar = progress
  else:
    pbar = tqdm(
      total=total, 
      desc=(progress if isinstance(progress, str) else None),
      disable=(not progress)
    )

  results = []

  exceptions = []

  def add_exception(greenlet):
    nonlocal exceptions
    try:
      greenlet.get()
    except Exception as err:
      exceptions.append(err)
  
  def updatefn(fn):
    def realupdatefn():
      res = fn()
      pbar.update(1)
      results.append(res)
    return realupdatefn

  pool = gevent.pool.Pool(concurrency)
  for fn in fns:
    greenlet = pool.spawn( updatefn(fn) )
    greenlet.link_exception(add_exception)

  pool.join()
  pool.kill()
  pbar.close()

  if exceptions:
    raise_multiple(exceptions)

  return results

def schedule_single_threaded_jobs(
  fns, progress=None, total=None
):
  if isinstance(progress, tqdm):
    pbar = progress
  else:
    pbar = tqdm(
      total=totalfn(fns, total), 
      disable=(not progress), 
      desc=(progress if isinstance(progress, str) else None)
    )

  with pbar:
    results = []
    for fn in fns:
      res = fn()
      pbar.update(1)
      results.append(res)
  return results 

def schedule_jobs(
    fns, concurrency=DEFAULT_THREADS, 
    progress=None, total=None, green=False
  ):
  """
  Given a list of functions, execute them concurrently until
  all complete. 

  fns: iterable of functions
  concurrency: number of threads (0: no threads)
  progress: Falsey (no progress), String: Progress + description
  total: If fns is a generator, this is the number of items to be generated.
  green: If True, use green threads.

  Return: list of results
  """
  if concurrency < 0:
    raise ValueError("concurrency value cannot be negative: {}".format(concurrency))
  elif (
    (isinstance(total, int) and total == 0)
    or (hasattr(fns, "__len__") and len(fns) == 0)
  ):
    return [] 
  elif (
    concurrency == 0 
    or (isinstance(total, int) and total <= 1) 
    or (hasattr(fns, "__len__") and len(fns) <= 1)
  ):
    return schedule_single_threaded_jobs(fns, progress, total)

  if isinstance(total, int):
    concurrency = min(concurrency, max(total, 1))
  elif hasattr(fns, "__len__"):
    concurrency = min(concurrency, max(len(fns), 1))

  if green:
    return schedule_green_jobs(fns, concurrency, progress, total)

  return schedule_threaded_jobs(fns, concurrency, progress, total)

# c/o https://stackoverflow.com/questions/12826291/raise-two-errors-at-the-same-time
def raise_multiple(errors):
  if not errors:
    return
  try:
    raise errors.pop()
  finally:
    raise_multiple(errors)


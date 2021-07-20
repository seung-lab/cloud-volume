import sys

from tqdm import tqdm

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

  pbar = tqdm(total=total, desc=progress, disable=(not progress))
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

  pbar = tqdm(total=total, desc=progress, disable=(not progress))
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

def schedule_jobs(
    fns, concurrency=DEFAULT_THREADS, 
    progress=None, total=None, green=False
  ):
  """
  Given a list of functions, execute them concurrently until
  all complete. 

  fns: iterable of functions
  concurrency: number of threads
  progress: Falsey (no progress), String: Progress + description
  total: If fns is a generator, this is the number of items to be generated.
  green: If True, use green threads.

  Return: list of results
  """
  if concurrency < 0:
    raise ValueError("concurrency value cannot be negative: {}".format(concurrency))
  elif concurrency == 0:
    return [ fn() for fn in tqdm(fns, disable=(not progress), desc=progress) ]

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


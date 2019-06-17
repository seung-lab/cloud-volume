import sys

from tqdm import tqdm

from .threaded_queue import ThreadedQueue, DEFAULT_THREADS

def schedule_threaded_jobs(
    fns, concurrency=DEFAULT_THREADS, 
    progress=None, total=None
  ):

  def wrap(fn):
    def wrapped(iface):
      return fn()
    return wrapped

  with ThreadedQueue(n_threads=concurrency, progress=progress) as tq:
    for fn in fns:
      tq.put(wrap(fn))

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
  
  def updatefn(fn):
    def realupdatefn():
      res = fn()
      pbar.update(1)
      results.append(res)
    return realupdatefn

  pool = gevent.pool.Pool(concurrency)
  for fn in fns:
    pool.spawn( updatefn(fn) )

  pool.join()
  pool.kill()
  pbar.close()

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
  if green:
    return schedule_green_jobs(fns, concurrency, progress, total)

  return schedule_threaded_jobs(fns, concurrency, progress, total)



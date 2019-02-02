import sys

from tqdm import tqdm

from .threaded_queue import ThreadedQueue, DEFAULT_THREADS

if sys.version_info[0] >= 3:
  import gevent.pool

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


if sys.version_info[0] < 3:
  schedule_jobs = schedule_threaded_jobs
else:
  schedule_jobs = schedule_green_jobs
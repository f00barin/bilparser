import time
import logging
import multiprocessing

def iterqueue(q, name='items', log_interval=0):

    start = time.time()

    def elapsed():
        count = time.time() - start
        unit = 's'
        if count < 10:
            count *= 1000
            unit = 'ms'
        return count, unit

    count = 0
    while True:
        try:
            yield q.get(timeout=0.3)
            count += 1
        except:
            logging.warning('processed %d %s in %d%s', count, name, *elapsed())
            break
        if log_interval and not count % log_interval:
            logging.info('processed %d %s in %d%s', count, name, *elapsed())

def launch(concurrency, sents, target, parser, mkargs=None, **kwargs):

    sent_q = multiprocessing.Queue()
    multiprocessing.Process(target=_enqueue, args=(sent_q, sents)).start()
    result_q = multiprocessing.Queue()
    for i in xrange(concurrency):
        extra_args = ()
        if callable(mkargs):
            extra_args = mkargs(i)
        args = (sent_q, result_q, parser) + extra_args
        multiprocessing.Process(target=target, args=args, kwargs=kwargs).start()
    return result_q

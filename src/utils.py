from collections import namedtuple

NetOutput = namedtuple('NetOutput', 'detection landmarks visibility pose gender')
OutputBatch = namedtuple('OutputBatch', 'detection landmarks visibility pose gender')
NetLosses = namedtuple('NetLosses', 'detection landmarks visibility pose gender total')
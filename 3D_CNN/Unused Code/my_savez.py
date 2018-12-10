import numpy as np
import tempfile
from sys import platform

class mySavez(object):
    @classmethod
    def __init__(self, file):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        # Import deferred for startup time improvement
        import tempfile
        import os

        if isinstance(file,str):
            if not file.endswith('.npz'):
                file = file + '.npz'

        compression = zipfile.ZIP_STORED

        zip = self.zipfile_factory(file, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
        os.close(fd)

        self.tmpfile = tmpfile
        self.zip = zip
        self.i = 0

    @classmethod
    def zipfile_factory(self, *args, **kwargs):
        import zipfile
        import sys
        if sys.version_info >= (2, 5):
            kwargs['allowZip64'] = True
        return zipfile.ZipFile(*args, **kwargs)

    @classmethod
    def savez(self, *args, **kwds):
        import os
        import numpy.lib.format as format

        namedict = kwds
        for val in args:
            key = 'arr_%d' % self.i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
            self.i += 1

        try:
            for key, val in namedict.items():
                fname = key + '.npy'
                fid = open(self.tmpfile, 'wb')
                try:
                    format.write_array(fid, np.asanyarray(val))
                    fid.close()
                    fid = None
                    self.zip.write(self.tmpfile, arcname=fname)
                finally:
                    if fid:
                        fid.close()
        finally:
            os.remove(self.tmpfile)

    @classmethod
    def close(self):
        self.zip.close()

# tmp = tempfile.TemporaryFile()
# f = my_savez(tmp)
# for i in range(10):
#     array = np.zeros(10)
#     f.savez(array)
# f.close()
#
# tmp.seek(0)
#
# tmp_read = np.load(tmp)
# # # print(tmp_read.files)
# for k in tmp_read.iteritems():
#      print(k[1])
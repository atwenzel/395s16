"""

This module provides functionality to easily browse, format, and export
multi-dimensional analysis outputs.  For example, this is particularly useful
for analyses where there are multiple metrics and combinations of input
variables or filters.  In total, each group of related data sets is termed a
bundle.

Specifically, this module has an interface to support adding and tagging
multiple data sets or distributions. Also, it supports multiple "views" (i.e.
plots to export), as well as inheritance between different views.

It should be possible to load multiple bundles to compare different metrics
side by side.

The entire repository of data sets/distributions can be serialized and easily
loaded for future analysis. An index file maintains all the tag information,
with a link to the data file that contains the individual distribution's data.

Gzipped pickle serialization is used. This allows us to use set and frozen set
structures, as well as dictionaries with non-string keys.

"""

import gzip, os
import cPickle as pickle

def coerceToUnicode(obj):
    if obj == None:
        return obj
    if type(obj) == unicode:
        return obj
    if type(obj) == str:
        return unicode(obj)
    if type(obj) == float:
        return obj
    if type(obj) == int:
        return obj
    if type(obj) == bool:
        return obj
    if type(obj) == frozenset:
        return frozenset([coerceToUnicode(x) for x in obj])
    if type(obj) == set:
        return set([coerceToUnicode(x) for x in obj])
    if type(obj) == list:
        return list([coerceToUnicode(x) for x in obj])
    if type(obj) == tuple:
        return tuple([coerceToUnicode(x) for x in obj])
    if type(obj) == dict:
        d = {}
        for k in obj:
            d[coerceToUnicode(k)] = coerceToUnicode(obj[k])
        return d
    raise RuntimeError("unhandled type: %s" % type(obj))

def getNamePath(path):
    if not os.path.isdir(path):
        raise RuntimeError("bundle does not exist at %s" % path)

    path, name = path.rsplit(os.sep, 1)

    if not name.startswith(bundle.prefix):
        raise RuntimeError("bundle does not exist at %s" % path)

    name = name[len(bundle.prefix):]

    return name, path

class bundle:

    prefix = "bundle-"
    dataPrefix = "data_"
    suffix = ".pickle.gz"
    indexFile = "index"
    

    def __init__(self, name, path=None):
        self.name = name

        if path is None: path = os.getcwd()
        self.path = os.sep.join([path, "%s%s" % (bundle.prefix, name)])

        if self._exists(): self._load() # try to load...  
        else: self._init()

    def _exists(self):
        if not os.path.isdir(self.path): return False
        # may need to add other checks here...
        return True

    def _init(self):
        os.mkdir(self.path)

        self.index = {}
        self._save()

    def _load(self):
        f = gzip.open(os.sep.join([self.path, "%s%s" % (bundle.indexFile, bundle.suffix)]))
        self.index = pickle.load(f)
        f.close()
    
    def _save(self):
        f = gzip.open(os.sep.join([self.path, "%s%s" % (bundle.indexFile, bundle.suffix)]), "w")
        pickle.dump(self.index, f)
        f.close()

    def add(self, data, **params):
        params = coerceToUnicode(params)

        fname = "%s%s" % (bundle.dataPrefix, "__".join(["%s_%s" % (k, params[k]) for k in sorted(params)]))

        # check if this key has already been added
        if fname in self.index:
            raise RuntimeError("key collision! %s" % fname)

        # save this data in a file
        f = gzip.open(os.sep.join([self.path, "%s%s" % (fname, bundle.suffix)]), "w")
        pickle.dump(data, f)
        f.close()

        # save in the index
        self.index[fname] = params

        self._save()

    def get(self, **params):
        params = coerceToUnicode(params)

        found = False
        for fname in self.index:
            if self.index[fname] == params:
                found=True
                break
        if not found:
            raise RuntimeError("unknown data for key: %s" % str(params))

        f = gzip.open(os.sep.join([self.path, "%s%s" % (fname, bundle.suffix)]))
        data = pickle.load(f)
        f.close()

        return data

    def list(self, *tags, **params):
        tags = coerceToUnicode(tags)
        params = coerceToUnicode(params)

        filtered = set(self.index)

        # select subset where these tags are all present
        filtered = [x for x in filtered if set(self.index[x]) & set(tags) == set(tags)]

        final = []
        for x in filtered:
            d = self.index[x]

            okay = True
            for needK in params:
                needV = params[needK]
                if needK not in d or d[needK] != needV:
                    okay = False
                    break
            if okay:
                final.append(d)

        return sorted(final)
    
    def delete(self, **params):
        params = coerceToUnicode(params)

        fname = "%s%s" % (bundle.dataPrefix, "__".join(["%s_%s" % (k, params[k]) for k in sorted(params)]))

        # make sure this key exists
        if fname not in self.index:
            raise RuntimeError("key does not exist! %s" % fname)

        # delete this file
        fpath = os.sep.join([self.path, "%s%s" % (fname, bundle.suffix)])
        os.remove(fpath)

        # remove from the index
        del self.index[fname]

        self._save()

    def update(self, data, **params):
        if self.list(**params):
            self.delete(**params)
        self.add(data, **params)

if __name__ == "__main__":
    x = [1,2,3]
    y = [4,5,6]
    z = [7,8,9]

    b = bundle("test")
    b.add(x, name="foo", val=5)
    b.add(y, name="bar", val=7, hello=True)
    b.add(z, name="baz")

    for x in b.list():
        print x
    print
    for x in b.list("name"):
        print x
    print
    for x in b.list("val"):
        print x
    print
    for x in b.list(val=5):
        print x

    print b.get(**b.list(val=5)[0])

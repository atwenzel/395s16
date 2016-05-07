"""Utility Scripts, mostly for local operations, reading configs, etc."""

#Global
import sys

#Local

def read_config(path):
    """Takes a relative path to a config file and returns a config dict. Assumes:
        1) Commented lines begin with '#'
        2) All lines are of format 'field=value'
        3) MUST HAVE space characters on either side of the '='
    """
    counter = 0
    cdict = {}
    try:
        clines = open(path, 'r').readlines()
    except IOError:
        print("Config doesn't exist: "+path)
        sys.exit(-1)
    for line in clines:
        if line[0] == '#' or line == "\n":
            counter += 1
            continue #ignore comments and blank lines
        parts = line.strip('\n').split(' = ')
        if len(parts) != 2:
            print("Bad config syntax in config: "+path+" at line "+str(counter))
            sys.exit(-1)
        cdict[parts[0]] = parts[1]
        counter += 1
    return cdict

def fix_json(badstr):
    """Takes a string that, other than bad quotes, is a valid JSON string, and replaces with double quotes"""
    newstr = ''
    for char in badstr:
        if char == "'":
            newstr += '"'
        else:
            newstr += char
    return newstr

if __name__ == "__main__":
    print(read_config('sample_config.dat'))

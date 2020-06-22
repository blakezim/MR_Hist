"""Config file spec system using YAML with validation"""

import yaml
import os.path


class Param(object):
    """
    Simple parameter class

    Should give a default value which is used for the skeleton file and is the
    default if not required and not provided by the user

    Also, children is either None or a dict containing other Params

    The rare and debug flags just control whether or not this option is shown
    in sample output
    """
    def __init__(self, default=None, required=False, rare=False, debug=False,
                 comment=None):
        self.default = default
        self.required = required
        self.comment = comment
        self.rare = rare  # NOTE: These are currently ignored entirely --JDH
        self.debug = debug

    def commentString(self):
        """Comment string for YAML"""
        s = ''
        if self.required or self.comment is not None:
            s += "  # "
        if self.required:
            s += "REQUIRED "
        if self.comment is not None:
            s += self.comment
        return s

    def decoratedString(self, level=0):
        """Output as YAML example"""
        s = ''

        # comment all lines if this is an optional param
        c = ''
        if self.required is False:
            c = '# '

        if isinstance(self.default, list):
            s += self.commentString()
            for l in self.default:
                s += "\n" + "    "*level + c + '- ' + str(l)
        elif isinstance(self.default, dict):
            s += self.commentString()
            for k,v in self.default.iteritems():
                s += "\n" + "    "*level + c + k + ': ' + str(v)
        else:
            s += str(self.default) + self.commentString()

        return s

    def __str__(self):
        return self.decoratedString()


def SpecToDict(spec):
    """Given a spec, generate a sample dict, as read by LoadYAMLDict()"""
    d = {}
    for k, v in spec.iteritems():
        if k[0] is '_':
            continue  # stuff with _ at the beginning is probably a hook
        elif isinstance(v, Param):
            d[k] = v.default
        elif isinstance(v, dict):
            d[k] = SpecToDict(v)
        else:
            raise Exception("Malformed config spec. " +
                            "Should be dict tree with Param leaves")
    return d


def DictKeysToAttributes(d):
    """Given a dict, convert key/value pairs to attributes"""

    if isinstance(d, dict):
        o = _Object()
        for k, v in d.iteritems():
            if k in o.__dict__:
                raise Exception("Key " + k + " exists as attribute." +
                                " Error in config file spec")
            else:
                o.__dict__[k] = DictKeysToAttributes(v)
    else:  # we're at a leaf
        return d

    return o


def SpecToConfig(spec):
    """Given a spec, generate a sample config object

    This is useful, for instance, in a test script where one might call this to
    get a config object, then customize a few parameters before running the
    test.
    """
    return DictKeysToAttributes(SpecToDict(spec))


def SpecToYAML(spec, level=0):
    """Convert a config spec to YAML"""
    lines = []
    for k, v in spec.iteritems():
        if k[0] is '_':
            continue  # stuff with _ at the beginning is probably a hook
        elif isinstance(v, Param):
            # Only way to do None is missing entry, so comment out if default
            # value is None
            com = "# " if v.default is None or not v.required else ""

            sv = v.decoratedString(level+1)
            lines.append("%s%s%s: %s" % ("    "*level, com, k, sv))
        elif isinstance(v, dict):
            lines.append("%s%s:" % ("    "*level, k))
            lines.append(SpecToYAML(v, level+1))
        else:
            raise Exception("Malformed config spec. " +
                            "Should be dict tree with Param leaves")
    return '\n'.join(lines)


def ConfigToYAML(spec, d, level=0):
    """Convert a config, given spec to YAML"""
    lines = []
    for k, v in spec.iteritems():
        if k[0] is '_':
            continue  # stuff with _ at the beginning is probably a hook
        elif k not in d.__dict__:
            raise Exception("Key %s missing.  Not validated?" % k)
        elif isinstance(v, Param):
            lines.append("%s%s: %s" % ("    "*level, k, d.__dict__[k]))
        elif isinstance(v, dict):
            lines.append("%s%s:" % ("    "*level, k))
            lines.append(ConfigToYAML(spec=spec[k], d=d.__dict__[k],
                                      level=level+1))
        else:
            raise Exception("Malformed config spec. " +
                            "Should be dict tree with Param leaves")
    return '\n'.join(lines)


class IncludeLoader(yaml.Loader):  # pylint: disable-msg=R0901
    """
    Subclassed yaml loader that supports the !include directive
    """
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(IncludeLoader, self).__init__(stream)

    def include(self, node):
        """Include another yaml at this node"""
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(os.path.expanduser(filename), 'r') as f:
            return yaml.load(f, Loader=IncludeLoader)

IncludeLoader.add_constructor('!include', IncludeLoader.include)

class IgnoreLoader(yaml.Loader):

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(IgnoreLoader, self).__init__(stream)

    def let_include_through(self, node):
        return self.construct_scalar(node)
    
IgnoreLoader.add_constructor(u'!include',IgnoreLoader.let_include_through)
        


class MissingKeyError(Exception):
    """Exception type for missing config params"""
    def __init__(self, value=None):
        super(MissingKeyError, self).__init__(self, value)
        self.value = value

    def __str__(self):
        return repr(self.value)


def ValidateDict(d, spec, prefix=""):
    """Validate a dict and insert defaults, conforming to spec"""
    if d is None:  # missing subsections should be handled gracefully
        # this happens, eg when all params in a section are optional
        d = {}

    for k, v in spec.iteritems():
        if k[0] is '_':
            continue  # stuff with _ at the beginning is probably a hook
        elif isinstance(v, dict):
            # do the recursive thing
            # NOTE: if your spec is a tree then everything below is required
            if k not in d.keys():
                raise MissingKeyError("Required subsection %s is missing" % k)
            else:
                if d[k] is None:  # handle missing sections
                    d[k] = {}
                ValidateDict(d[k], spec[k], prefix=prefix + "." + k)
        elif isinstance(v, Param):
            if k not in d.keys():
                if v.required:
                    raise MissingKeyError("Required param %s is missing" % k)
                else:
                    d[k] = v.default
            else:  # they supplied it, we're good
                pass
        else:
            raise Exception("Malformed config spec at key %s.%s" % (prefix, k)
                            + ". Should be dict tree with Param leaves")

    # warn about extra keys (could be misspelled params with default values)
    for k in set(d.keys())-set(spec.keys()):
        # skip stuff starting with underscore
        if k[:1] is not "_":
            print("Warning: config key " + prefix + "." + k +
                  " not covered by spec.  Ignoring...")
        # remove this cruft from the final config
        del d[k]


class _Object(object):
    """Just an empty object that we can set attributes on"""
    def __init__(self):
        pass

    def __setattr__(self, name, value):
        '''Prevent new attributes from being added by the 'normal' object.name = value syntax
        Creating attribues by accessing the __dict__ directly still works.'''
        if not hasattr(self, name):
            raise AttributeError(
                '{} instance has no attribute defined in the specification for {!r} -- check your spelling?'.format(
                    type(self).__name__, name))
        object.__setattr__(self, name, value)


def LoadYAMLDict(configFile, include=True):
    """Load a yaml file and insert default values"""
    stream = open(configFile, 'r')
    if include==False:
        return yaml.load(stream, Loader=IgnoreLoader)
    else:
        return yaml.load(stream, Loader=IncludeLoader)

class MissingConfigError(Exception):
    """Exception type for missing config file"""
    def __init__(self, value=None):
        super(MissingConfigError, self).__init__(self, value)
        self.value = value

    def __str__(self):
        return repr(self.value)


def RunValidationHooks(c, spec):
    """Recursively search for hooks and run them"""
    for k, v in spec.iteritems():
        if k is "_validation_hook":
            spec["_validation_hook"](c)
        elif isinstance(v, dict):
            RunValidationHooks(c.__dict__[k], v)


def MkConfig(d, spec):
    """
    Given a dict and a spec, validate and return a config object

    If you do not use yaml config files, or are calling functions from a test
    script, this is the preferred way to set up configs.
    """

    if not 'scales' in d.keys():
        ValidateDict(d, spec)

        # convert dict keys to Object attributes (safely)
        c = DictKeysToAttributes(d)

        # run hooks if supplied by spec
        RunValidationHooks(c, spec)

        return c
    else:
        # multiscale config. must validate each section
        o = _Object()
        o.__dict__['scales'] = d['scales']

        for si in d['scales']:
            subname = 'paramsScale'+str(si)
            if not subname in d.keys():
                raise Exception(subname + ' config section missing')
            # each of these sections must be valid configs
            o.__dict__[subname] = MkConfig(d[subname], spec)
        # HACK: there should be a better way to handle multiscale vs single
        o.__dict__['study'] = o.__dict__['paramsScale'+str(d['scales'][0])].study
        return o


def Load(spec, argv, extraText=''):
    """Load YAML file, validate it, given spec (inserting defaults) and convert
    to attributes

    The optional argument extraText is printed in addition to the sample config,
    if no config is provided
    """
    if len(argv) < 2:
        print('# Usage: ' + argv[0] + ' <config.yaml>')
        print('# Below is a sample config YAML file')
        print('# Optional parameters are commented, with default values shown')
        print(SpecToYAML(spec))
        # print extraText
        if '_resource' in spec:
            # this won't be printed by default, but let's keep track
            print("_resource: " + spec['_resource'])

        raise MissingConfigError()

    d = LoadYAMLDict(argv)

    # TODO: process overrides (given on the cmdline as key=value pairs)

    return MkConfig(d, spec)


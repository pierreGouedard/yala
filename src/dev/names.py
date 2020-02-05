# Global imports

# Local imports


class KVName(object):
    def __init__(self, obj=None, sep=',', equ='='):
        """
        Key-Value name manager.
        This class is both a key value store and an interface for serialized strings of the form:
        "key1=value1,key2=value2,...".
        It is a good thing to feed this class with string keys and string values, so that serialization/deserialization
        does not loose information. Also note that serialization will always reorder keys alphabetically.
        :param (dict|str) obj: Either a dict or a string to initialize the key-value store.
                               If None, the store will be empty.
        :param str sep: String separator
        """
        assert(sep != equ, 'separator and equlaty must be different')

        self._splits = {}
        self._sep = sep
        self._equ = equ

        if obj is None:
            pass
        elif isinstance(obj, str) or isinstance(obj, unicode):
            self.set_from_string(obj)
        elif isinstance(obj, dict):
            self.set_from_dict(obj)
        else:
            raise TypeError('Input object of KVName of type {} was not recognized. '
                            'Try KVName.from_string or KVName.from_dict.'.format(type(obj)))

    @staticmethod
    def from_string(s, sep=',', equ='='):
        """
        Create a KVName from a string.
        :param str s:
        :param str sep:
        :rtype: KVName
        """
        out = KVName(sep=sep, equ=equ)
        out.set_from_string(s)
        return out

    @staticmethod
    def from_dict(d, sep=',', equ='='):
        """
        Create a KVName from a dict.
        :param dict d:
        :param str sep:
        :rtype: KVName
        """
        out = KVName(sep=sep, equ=equ)
        out.set_from_dict(d)
        return out

    def set_from_string(self, s):
        """
        Set the key value store from a string.
        :param str s:
        :return: The class itself
        :rtype: KVName
        """
        try:
            self._splits = dict([tuple(k.split(self._equ)) for k in s.split(self._sep)])
        except Exception as e:
            raise e.__class__('Could not parse "{}" (separator="{}", equality="{}"): {}'.format(s, self._sep, self._equ,
                                                                                                e.args))
        return self

    def set_from_dict(self, d):
        """
        Set the key value store from a dict.
        :param dict d:
        :return: The class itself
        :rtype: KVName
        """
        self._splits = d.copy()
        return self

    def to_string(self):
        """
        Serialize the key value store to a string.
        :rtype: str
        """
        name = self._sep.join(['{}{}{}'.format(k, self._equ, v) for k, v in sorted(self._splits.items())])
        return name

    def to_dict(self):
        """
        Dump the key value store as a dict.
        :rtype: dict
        """
        return self._splits.copy()

    def keys(self):
        """
        Current keys in the key value store (alphabetically ordered).
        :rtype: tuple
        """
        return tuple(sorted(self._splits.keys()))

    def values(self):
        """
        Current values in the key value store.
        Values come in the same orders as keys.
        :rtype: tuple
        """
        return tuple(self._splits[k] for k in self.keys())

    def items(self):
        """
        Current items in the key value store.
        :rtype: list[(str,str)]
        """
        return zip(self.keys(), self.values())

    def drop_keys(self, *levels):
        """
        Drop keys from the key value store
        :param str levels:
        :return: The class itself
        :rtype: KVName
        """
        for l in levels:
            if l not in self._splits:
                raise ValueError('Could not drop unknown level "{}"'.format(l))
            del self._splits[l]
        return self

    def __setitem__(self, key, value):
        """
        :param str key:
        :param str value:
        """
        self._splits[key] = value

    def __getitem__(self, item):
        """
        :param str item:
        :rtype: str
        """
        return self._splits[item]

    def set_item(self, key, value):
        self[key] = value
        return self

    def copy(self):
        """
        Make a copy of the class instance
        :rtype: KVName
        """
        return KVName.from_dict(self._splits, sep=self._sep)

    def change_sep(self, new_sep):
        """
        Change the string separator.
        :param str new_sep:
        :return: The class itself
        :rtype: KVName
        """
        self._sep = new_sep
        return self

    def change_equ(self, new_equ):
        """
        Change the string separator.
        :param str new_sep:
        :return: The class itself
        :rtype: KVName
        """
        self._equ = new_equ
        return self

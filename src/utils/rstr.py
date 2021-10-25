# From: https://stackoverflow.com/questions/51477200/how-to-use-logger-to-print-a-list-in-just-one-line-in-python

class rstr:
    """Wrapper to recursively str()ise a list or tuple or set. The work is only
    performed in the __str__ call so these objects 'cost' very little if they
    are never actually used (e.g. as in a logger.debug() argument when the
    message is never output."""

    def __init__(self, seq):
        """Creates an rstr instance which will string-ise the argument when
        called."""
        self._seq = seq

    def __str__(self):
        """String-ise and return the argument passed to the constructor."""
        if isinstance(self._seq, list):
            return "[" + self._str_items() + "]"
        elif isinstance(self._seq, tuple):
            return "(" + self._str_items() + ")"
        elif isinstance(self._seq, set):
            return "{" + self._str_items() + "}"
        else:
            return str(self._seq)

    def _str_items(self):
        """Returns the string-ised forms of the items in the argument passed to
        the constructor - no start/end brackets/braces."""
        return ", ".join(map(str, self._seq))

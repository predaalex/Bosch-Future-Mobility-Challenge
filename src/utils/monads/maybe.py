class Maybe:
    def __init__(self, value, trace=[]):
        self._value = value
        self._trace = trace

    def bind(self, func):
        if self._value is None:
            return Maybe(None)
        else:
            return Maybe(func(self._value))

    def orElse(self, default):
        if self._value is None:
            return Maybe(default)
        else:
            return self

    def unwrap(self):
        return self._value

    def __or__(self, other):
        return Maybe(self._value or other._value)

    def __str__(self):
        if self._value is None:
            return 'Nothing'
        else:
            return 'Just {}'.format(self._value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Maybe):
            return self._value == other._value
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __bool__(self):
        return self._value is not None

if __name__ == "__main__":
    def add_one(x):
        return x + 1

    def double(x):
        return x * 2

    result = Maybe(3) \
        .bind(add_one) \
        .bind(add_one) \
        .bind(add_one) \
        .bind(add_one) \
        .bind(add_one) \
        .bind(add_one) \
        .bind(double)
    print(result)  # Just 12

    result = Maybe(None).bind(add_one).bind(double)
    print(result)  # Nothing

    #declarative programming
    result = Maybe(None).bind(add_one).bind(double).orElse(10).bind(add_one).bind(double)
    print(result)  # Just 11

    result = Maybe(None) | Maybe(1)
    print(result) # Just 1
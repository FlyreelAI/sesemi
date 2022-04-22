from sesemi.registries import CallableRegistry


def test_callable_registry():
    cr = CallableRegistry()

    def test():
        pass

    fn1 = cr(test)

    assert fn1 is test
    assert cr["test"] is fn1

    fn2 = cr("key")(test)
    assert fn2 is test
    assert cr["key"] is test

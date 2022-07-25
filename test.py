class Test:

    def __getattr__(self, key):
        print(key)


t = Test()


t.a.b

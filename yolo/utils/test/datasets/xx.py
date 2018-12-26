
class A(object):
    def __init__(self, a):
        self._a = a
    def say_hi(self):
        print 'hi'

class B(A):
    def __init__(self, b):
        #super(B, self).__init__(1)
        self.b = b

x = B(2)
print dir(x)
x.say_hi()
exit(0)
print x.b
print x.a

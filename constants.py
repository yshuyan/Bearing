class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:  # 判断是否已经被赋值，如果是则报错
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():  # 判断所赋值是否是全部大写，用来做第一次赋值的格式判断，也可以根据需要改成其他判断条件
            raise self.ConstCaseError(
                'const name "%s" is not all supercase' % name)

        self.__dict__[name] = value


const = Const()
const.MOTOR_LOAD_LEVELS = 4
const.SLIDING_WINDOW_LENGTH = 20

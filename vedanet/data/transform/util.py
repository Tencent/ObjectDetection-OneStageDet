#
#   Lightnet related data processing
#   Utilitary classes and functions for the data subpackage
#   Copyright EAVISE
#

from abc import ABC, abstractmethod

__all__ = ['Compose']


class Compose(list):
    """ This is lightnet's own version of :class:`torchvision.transforms.Compose`.

    Note:
        The reason we have our own version is because this one offers more freedom to the user.
        For all intends and purposes this class is just a list.
        This `Compose` version allows the user to access elements through index, append items, extend it with another list, etc.
        When calling instances of this class, it behaves just like :class:`torchvision.transforms.Compose`.

    Note:
        I proposed to change :class:`torchvision.transforms.Compose` to something similar to this version,
        which would render this class useless. In the meanwhile, we use our own version
        and you can track `the issue`_ to see if and when this comes to torchvision.

    Example:
        >>> tf = ln.data.transform.Compose([lambda n: n+1])
        >>> tf(10)  # 10+1
        11
        >>> tf.append(lambda n: n*2)
        >>> tf(10)  # (10+1)*2
        22
        >>> tf.insert(0, lambda n: n//2)
        >>> tf(10)  # ((10//2)+1)*2
        12
        >>> del tf[2]
        >>> tf(10)  # (10//2)+1
        6

    .. _the issue: https://github.com/pytorch/vision/issues/456
    """
    def __call__(self, data):
        for tf in self:
            data = tf(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        for tf in self:
            format_string += '\n  {tf}'
        format_string += '\n]'
        return format_string


class BaseTransform(ABC):
    """ Base transform class for the pre- and post-processing functions.
    This class allows to create an object with some case specific settings, and then call it with the data to perform the transformation.
    It also allows to call the static method ``apply`` with the data and settings. This is usefull if you want to transform a single data object.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, data):
        return self.apply(data, **self.__dict__)

    @classmethod
    @abstractmethod
    def apply(cls, data, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        return data


class BaseMultiTransform(ABC):
    """ Base multiple transform class that is mainly used in pre-processing functions.
    This class exists for transforms that affect both images and annotations.
    It provides a classmethod ``apply``, that will perform the transormation on one (data, target) pair.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @abstractmethod
    def __call__(self, data):
        return data

    @classmethod
    def apply(cls, data, target=None, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            target (optional): ground truth for that data; Default **None**
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        obj = cls(**kwargs)
        res_data = obj(data)

        if target is None:
            return res_data

        res_target = obj(target)
        return res_data, res_target

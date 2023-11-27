import argparse
from typing import TypeVar, Generic, cast

Value = TypeVar("Value")


class Arg(Generic[Value]):
    value: Value

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __set_name__(self, owner, name):
        owner.arguments.append(name)

    def __get__(self, instance, owner) -> Value:
        return self.value


class TypedNamespace(argparse.Namespace):
    arguments = []
    @property
    def descriptors(self):
        return [(arg, self.__class__.__dict__[arg]) for arg in self.arguments]


Namespace = TypeVar("Namespace", bound=argparse.Namespace)


class ParseArger(argparse.ArgumentParser, Generic[Namespace]):
    def __init__(self, *args, namespace: Namespace, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = namespace
        if namespace is not None:
            for name, descriptor in namespace.descriptors:
                self.add_argument(f"--{name.replace('_', '-')}", *descriptor.args, **descriptor.kwargs)

    def parse_args(self, args=None) -> Namespace:
        return cast(Namespace, super().parse_args(args=args, namespace=self.namespace))

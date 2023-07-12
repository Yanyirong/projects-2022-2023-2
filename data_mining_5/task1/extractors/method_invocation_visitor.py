from .visitor import Visitor
from typing import List


class MethodInvocationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.method_invocation_list = []

    def get_method_invocations(self, code: str) -> List[str]:


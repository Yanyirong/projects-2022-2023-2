from .visitor import Visitor
from typing import List


class MethodDeclarationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.method_names = []

    def get_method_names(self, code: str) -> List[str]:


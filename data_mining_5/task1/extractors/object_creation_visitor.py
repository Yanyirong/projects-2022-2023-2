from .visitor import Visitor
from typing import List


class ObjectCreationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.object_creation_list = []

    def get_object_creations(self, code: str) -> List[str]:
        # class_body = root.children[0].children[3]
        # for child in class_body.children:
        #     if child.type == 'method_declaration':
        #         self._get_object_creation(child)
        # return self.object_creation_list
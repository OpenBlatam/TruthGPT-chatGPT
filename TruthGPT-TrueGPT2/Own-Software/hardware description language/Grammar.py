
class FINENode:
    def __init__(self, HDLNode):
        self.name = name
        self.node = node

class HDLNode:
    def __init__(self):
        pass

class ModuleNode(HDLNode):
    def __init__(self, name, inputs, outputs, statements):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.statements = statements

class InputNode(HDLNode):
    def __init__(self, name, bitwidth):
        self.name = name
        self.bitwidth = bitwidth

class OutputNode(HDLNode):
    def __init__(self, name, bitwidth):
        self.name = name
        self.bitwidth = bitwidth

class StatementNode(HDLNode):
    def __init__(self):
        pass

class AssignNode(StatementNode):
    def __init__(self, target, value):
        self.target = target
        self.value = value

class BinaryOpNode(HDLNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class IdentifierNode(HDLNode):
    def __init__(self, name):
        self.name = name

class LiteralNode(HDLNode):
    def __init__(self, value):
        self.value = value

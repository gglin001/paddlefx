from __future__ import annotations

import collections
import copy
import dis
import functools
import logging
import operator
import types

from typing import Any, NamedTuple

import paddle
import paddle.nn

from .bytecode_transformation import (
    Instruction,
    create_call_function,
    create_instruction,
)
from .codegen import PyCodegen
from .output_graph import OutputGraph, unique_id
from .proxy import Attribute, Proxy
from .resume_execution import ContinueExecutionCache

# TODO: better code organization


def _binary_constructor(op_name: str):
    def _binary(self, inst: Instruction):
        op = getattr(operator, op_name)
        args = self.popn(2)
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    return _binary


def _unary_constructor(op_name: str):
    def _unary(self, inst: Instruction):
        op = getattr(operator, op_name)
        res = self.output.create_node('call_function', op, self.pop(), {})
        self.push(res)

    return _unary


def _not_implemented(op_name):
    def _not_impl(self, inst):
        raise NotImplementedError()

    return _not_impl


BINARY_MAPPER = {
    'add': 'BINARY_ADD',
    'sub': 'BINARY_SUBTRACT',
    'mul': 'BINARY_MULTIPLY',
    'floordiv': 'BINARY_FLOOR_DIVIDE',
    # NOTE: in fact, paddle doesn't support floor_divide
    'truediv': 'BINARY_TRUE_DIVIDE',
    'mod': 'BINARY_MOD',
    'pow': 'BINARY_POWER',
    'matmul': 'BINARY_MATMUL',
    'getitem': 'BINARY_GETITEM',
    'lshift': 'BINARY_LSHIFT',
    'rshift': 'BINARY_RSHIFT',
    'iadd': 'INPLACE_ADD',
    'ifloordiv': 'INPLACE_FLOOR_DIVIDE',
    'imod': 'INPLACE_MOD',
    'imul': 'INPLACE_MULTIPLY',
    'imatmul': 'INPLACE_MATRIX_MULTIPLY',
    'ipow': 'INPLACE_POWER',
    'isub': 'INPLACE_SUBTRACT',
    'itruediv': 'INPLACE_TRUE_DIVIDE',
}

UNARY_MAPPER = {'not_': 'UNARY_NOT', 'inv': 'UNARY_INVERT'}

NOT_IMPLEMENT = {
    'and_': 'BINARY_AND',
    'or_': 'BINARY_OR',
    'xor': 'BINARY_XOR',
    'iand': 'INPLACE_AND',
    'ior': 'INPLACE_OR',
    'ixor': 'INPLACE_XOR',
}


OP_MAPPER = [BINARY_MAPPER, UNARY_MAPPER, NOT_IMPLEMENT]
CONSTRUCTOR = [_binary_constructor, _unary_constructor, _not_implemented]


def break_graph_if_unsupported(*, push):
    def decorator(inner_fn):
        @functools.wraps(inner_fn)
        def wrapper(self: InstructionTranslatorBase, inst: Instruction):
            state = self.copy_graphstate()
            try:
                return inner_fn(self, inst)
            except NotImplementedError as exc:
                logging.debug(
                    "break_graph_if_unsupported triggered compile", exc_info=True
                )

            self.restore_graphstate(state)

            self.output.compile_subgraph(self)

            # append raw instruction
            assert inst.target is None
            inst_copy = copy.copy(inst)
            inst_copy.exn_tab_entry = None
            self.output.add_output_instructions([inst_copy])

            stack_effect = dis.stack_effect(inst.opcode, inst.arg)
            self.popn(push - stack_effect)

            for _ in range(push):
                self.push(None)

            self.output.add_output_instructions(
                self.create_call_resume_at(self.next_instruction)
            )

        return wrapper

    return decorator


class InstructionTranslatorGraphState(NamedTuple):
    # output: OutputGraphState
    symbolic_locals: dict[str, Any]
    # stack: List[VariableTracker]
    # block_stack: List[BlockStackEntry]
    block_stack: list[Any]
    instruction_pointer: int | None
    current_instruction: Instruction
    next_instruction: Instruction | None
    # lineno: int


class InstructionTranslatorBase:
    def __init__(
        self,
        *,
        instructions: list[Instruction],
        frame: types.FrameType,
        output: OutputGraph,
    ):
        self.instructions: list[Instruction] = instructions
        self.frame: types.FrameType = frame
        self.output: OutputGraph = output

        self.f_locals = {}
        self.f_globals: dict[str, Any] = frame.f_globals
        self.f_code = frame.f_code

        self.stack = []
        for k, _ in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)

        self.instruction_pointer = 0
        self.current_instruction: Instruction = create_instruction("NOP")
        self.next_instruction: Instruction | None = None

    def pop(self):
        return self.stack.pop()

    def push(self, item):
        return self.stack.append(item)

    def popn(self, n: int, reverse=True):
        assert n >= 0
        if not n:
            return []
        if reverse:
            return list(reversed([self.pop() for _ in range(n)]))
        else:
            return [self.pop() for _ in range(n)]

    def call_function(self, fn, args, kwargs):
        # TODO: implement InlineTranslator
        is_custom_call = False
        for arg in args:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break
        for arg in kwargs:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break

        if fn is print:
            raise NotImplementedError(f"print is not supported")
            # self.push(None)
        elif isinstance(fn, Attribute):
            self.push(fn(*args, **kwargs))
        elif fn is isinstance:
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.push(res)
        elif fn.__module__.startswith("paddle"):
            if hasattr(fn, "forward"):
                fn = fn.forward
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.push(res)
        elif is_custom_call:
            raise NotImplementedError(f"custom_call is not supported")
        else:
            raise NotImplementedError(f"call function {fn} is not supported")

    def LOAD_GLOBAL(self, inst: Instruction):
        name = inst.argval
        if name in self.frame.f_globals:
            self.push(self.frame.f_globals[name])
        elif name in self.frame.f_builtins:
            self.push(self.frame.f_builtins[name])
        else:
            raise Exception(f"name '{name}' is not found")

    def POP_JUMP_IF_FALSE(self, inst: Instruction):
        pass

    def POP_JUMP_IF_TRUE(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        value = inst.argval
        self.push(value)

    def LOAD_ATTR(self, inst: Instruction):
        obj = self.pop()
        if isinstance(obj, Proxy) and obj.node.name.startswith("self"):
            res = self.output.create_node('get_param', inst.argval)
            self.push(res)
        elif hasattr(obj, inst.argval):
            value = getattr(obj, inst.argval)
            self.push(value)
        else:
            self.push(None)

    def LOAD_METHOD(self, inst: Instruction):
        target = self.pop()
        fn = getattr(target, inst.argval)
        self.push(fn)

    def CALL_METHOD(self, inst: Instruction):
        args = self.popn(inst.argval)
        fn = self.pop()
        if isinstance(fn, Attribute):
            fn_name = repr(fn)
            if fn_name.startswith("self"):
                res = self.output.create_node('call_module', fn.attr, args, {})
            else:
                res = fn(*args)
            self.push(res)
        else:
            # TODO(zrr1999) other class should be handled separately.
            if hasattr(fn, "forward"):
                fn = fn.forward
            if fn is not None:
                res = self.call_function(fn, args, {})
                self.push(res)
            else:
                self.push(None)

    @break_graph_if_unsupported(push=1)
    def CALL_FUNCTION(self, inst: Instruction):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    def CALL_FUNCTION_KW(self, inst: Instruction):
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        args, kwargs = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs))
        self.call_function(fn, args, kwargs)

    def BUILD_TUPLE(self, inst):
        items = self.popn(inst.argval)
        self.push(tuple(items))

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        self.push(items)

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        result = dict()
        for k, v in zip(items[::2], items[1::2]):
            result[k] = v
        assert len(result) == len(items) / 2
        self.push(result)

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.pop()
        values = self.popn(inst.argval)
        self.push(dict(zip(keys, values)))

    def BINARY_SUBSCR(self, inst):
        idx = self.pop()
        root = self.pop()
        if isinstance(root, Proxy):
            res = root[idx]
        else:
            res = self.output.create_node('call_method', "__getitem__", [root, idx], {})
        self.push(res)

    def STORE_SUBSCR(self, inst):
        value = self.pop()
        idx = self.pop()
        root = self.pop()
        self.output.create_node('call_method', "__setitem__", [root, idx, value], {})

    def POP_TOP(self, inst: Instruction):
        value = self.pop()

    def STORE_FAST(self, inst: Instruction):
        self.f_locals[inst.argval] = self.pop()

    def LOAD_FAST(self, inst: Instruction):
        self.push(self.f_locals[inst.argval])

    def RETURN_VALUE(self, inst: Instruction):
        self.output.compile_subgraph(self)
        self.output.add_output_instructions(
            [
                create_instruction("RETURN_VALUE"),
            ]
        )

    def COMPARE_OP(self, inst: Instruction):
        op_mapper = {
            '>': 'gt',
            '<': 'lt',
            '>=': 'ge',
            '<=': 'le',
            '==': 'eq',
            '!=': 'ne',
            'is': 'is_',
            'is not': 'is_not',
        }
        op = getattr(operator, op_mapper[inst.argval])
        args = self.popn(2)
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    # note: python3.9+
    def IS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.popn(2)
        if invert:
            op = operator.is_
        else:
            op = operator.is_not
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    def CONTAINS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.popn(2)
        if invert:
            op = operator.contains
        else:
            op = lambda a, b: b not in a
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    def copy_graphstate(self) -> InstructionTranslatorGraphState:
        """Create a checkpoint of the current state by copying everything."""
        return InstructionTranslatorGraphState(
            # self.output.copy_graphstate(),
            collections.OrderedDict(self.f_locals),
            list(self.stack),
            # list(self.block_stack),
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            # self.lineno,
        )

    def restore_graphstate(self, state: InstructionTranslatorGraphState):
        """Restore a checkpoint created by self.copy_graphstate()"""
        (
            # output_state,
            self.f_locals,
            self.stack,
            # self.block_stack,
            self.instruction_pointer,
            self.current_instruction,
            self.next_instruction,
            # self.lineno,
        ) = state
        # self.output.restore_graphstate(output_state)


for mapper, constructor in zip(OP_MAPPER, CONSTRUCTOR):
    for op_name, func_name in mapper.items():
        func = constructor(op_name)
        func = types.FunctionType(
            func.__code__, globals(), None, None, func.__closure__
        )
        setattr(InstructionTranslatorBase, func_name, func)


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        *,
        instructions: list[Instruction],
        frame: types.FrameType,
        code_options: dict,
        compiler_fn: Any,
    ):
        output = OutputGraph(
            f_globals=frame.f_globals,
            code_options=code_options,
            compiler_fn=compiler_fn,
        )
        super().__init__(
            instructions=instructions,
            frame=frame,
            output=output,
        )

    def step(self):
        inst = self.instructions[self.instruction_pointer]
        self.current_instruction = inst
        self.instruction_pointer += 1

        if self.instruction_pointer < len(self.instructions):
            self.next_instruction = self.instructions[self.instruction_pointer]
        else:
            self.instruction_pointer = None
            self.next_instruction = None

        try:
            if not hasattr(self, inst.opname):
                raise Exception(f"missing: {inst.opname}")
            print(f"== TRACEING {inst}")
            getattr(self, inst.opname)(inst)

            return inst.opname != "RETURN_VALUE"
        except NotImplemented as exc:
            # TODO: create graph break
            raise exc

    def run(self):
        try:
            while (
                self.instruction_pointer is not None
                and not self.output.should_exit
                and self.step()
            ):
                pass
        except:
            raise

    def create_call_resume_at(self, inst: Instruction):
        self.instruction_pointer = None

        if inst.opname == "RETURN_VALUE":
            return [create_instruction("RETURN_VALUE")]

        argnames = tuple(k for k in self.f_locals.keys())

        cg = PyCodegen(self)

        stack_len = len(self.stack)
        nargs = stack_len + len(argnames)

        name = unique_id(f"__resume_at_{inst.offset}")

        # TODO: add block_stack
        self.block_stack = []
        new_code: types.CodeType = ContinueExecutionCache.lookup(
            self.f_code,
            inst.offset,  # self.lineno,
            inst.starts_line,
            tuple(b.target.offset for b in self.block_stack),
            stack_len,
            argnames,
            tuple(b.resume_fn() for b in self.block_stack),
            tuple([]),
        )

        print(f"new_code:")
        import dis

        [print(x) for x in dis.get_instructions(new_code)]

        self.output.install_global(
            name, types.FunctionType(new_code, self.f_globals, name)
        )
        cg.extend_output(cg.load_function_name(name, True, stack_len))

        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(create_call_function(nargs, False))
        cg.append_output(create_instruction("RETURN_VALUE"))
        return cg.get_instructions()

from __future__ import annotations

import dis

from typing import TYPE_CHECKING

from .bytecode_transformation import (
    create_instruction,
    create_load_global,
    create_rot_n,
)
from .translator import Instruction
from .utils import rot_n_helper

if TYPE_CHECKING:
    from .translator import Instruction, InstructionTranslatorBase


class PyCodegen:
    def __init__(
        self,
        tx: InstructionTranslatorBase = None,
    ):
        self.tx = tx
        self.code_options = self.tx.output.code_options

        self._output: list[Instruction] = []

    def append_output(self, inst):
        assert isinstance(inst, Instruction)
        self._output.append(inst)

    def extend_output(self, insts):
        assert all(isinstance(x, Instruction) for x in insts)
        self._output.extend(insts)

    def make_call_generated_code(self, fn_name: str):
        load_function = Instruction(
            opcode=dis.opmap["LOAD_GLOBAL"],
            opname="LOAD_GLOBAL",
            arg=False,
            argval=fn_name,
        )
        self.extend_output([load_function])
        self.tx.output.update_co_names(fn_name)

        placeholders = self.tx.output.placeholders
        for x in placeholders:
            load_fast = Instruction(
                opcode=dis.opmap["LOAD_FAST"],
                opname="LOAD_FAST",
                arg=None,
                argval=x.name,
            )
            self.extend_output([load_fast])

        call_function = Instruction(
            opcode=dis.opmap["CALL_FUNCTION"],
            opname="CALL_FUNCTION",
            arg=len(placeholders),
            argval=None,
        )
        self.extend_output([call_function])

    def create_load(self, name):
        # if name in self.cell_and_freevars():
        #     return create_instruction("LOAD_DEREF", argval=name)
        assert name in self.code_options["co_varnames"], f"{name} missing"
        return create_instruction("LOAD_FAST", argval=name)

    def create_load_global(self, name, push_null, add=False):
        if add:
            self.tx.output.update_co_names(name)
        assert name in self.code_options["co_names"], f"{name} not in co_names"
        return create_load_global(name, push_null)

    def load_function_name(self, fn_name, push_null, num_on_stack=0):
        """Load the global fn_name on the stack num_on_stack down."""
        output = []
        output.extend(
            [
                self.create_load_global(fn_name, False, add=True),
                *self.rot_n(num_on_stack + 1),
            ]
        )
        return output

    def rot_n(self, n):
        try:
            return create_rot_n(n)
        except AttributeError:
            # desired rotate bytecode doesn't exist, generate equivalent bytecode
            return [
                create_instruction("BUILD_TUPLE", arg=n),
                self._create_load_const(rot_n_helper(n)),
                *create_rot_n(2),
                create_instruction("CALL_FUNCTION_EX", arg=0),
                create_instruction("UNPACK_SEQUENCE", arg=n),
            ]

    def get_instructions(self):
        return self._output

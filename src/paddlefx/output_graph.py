from __future__ import annotations

import dis
import itertools
import logging

from typing import TYPE_CHECKING, Any

import paddle
import paddle.nn

from .bytecode_transformation import Instruction, unique_id
from .graph_layer import GraphLayer
from .symbolic_trace import Tracer

if TYPE_CHECKING:
    from .translator import Instruction, InstructionTranslatorBase, unique_id


class OutputGraph(Tracer):
    def __init__(
        self,
        *,
        f_globals: dict[str, Any],
        code_options: dict,
        compiler_fn: Any,
    ):
        super().__init__()

        self.f_globals = f_globals
        self.code_options = code_options
        self.compiler_fn = compiler_fn
        self.should_exit = False

        self.output_instructions: list[Instruction] = []

    @property
    def placeholders(self):
        r = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                r.append(node)
                continue
            break
        return r

    def new_var(self, name="tmp"):
        existing = set(self.code_options["co_varnames"])
        for i in itertools.count():
            var = f"___{name}_{i}"
            if var not in existing:
                self.code_options["co_varnames"] += (var,)
                return var

    def install_global(self, name, value) -> None:
        self.f_globals[name] = value

    def update_co_names(self, name: str):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)

    def add_output_instructions(self, prefix: list[Instruction]) -> None:
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def call_user_compiler(self, gl):
        compiled_fn = self.compiler_fn(gl)
        return compiled_fn

    def compile_and_call_fx_graph(self, tx, rv, root):
        from .codegen import PyCodegen
        from .eval_frame import disable

        self.create_node("output", "output", tuple(x for x in rv), {})

        gl = GraphLayer(root, self.graph)

        compiled_fn = self.call_user_compiler(gl)
        compiled_fn = disable(compiled_fn)

        name = unique_id("__compiled_fn")

        logging.debug(f"{name} - gl.src:\n{gl.src}")

        logging.debug(f"{name}:")
        [logging.debug(x) for x in list(dis.get_instructions(compiled_fn))]
        logging.debug(f"")

        logging.debug(f"{name}.fn:")
        [logging.debug(x) for x in list(dis.get_instructions(compiled_fn.fn))]
        logging.debug(f"")

        self.install_global(name, compiled_fn)

        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    def compile_subgraph(
        self,
        tx: InstructionTranslatorBase,
    ):
        from .codegen import PyCodegen

        stack_values = list(tx.stack)

        if not (root := tx.frame.f_locals.get('self', None)):
            root = paddle.nn.Layer()

        instructions = []

        instructions.extend(
            self.compile_and_call_fx_graph(
                tx,
                list(reversed(stack_values)),
                root,
            )
        )

        print(f"== stack_values: {stack_values}")

        graph_output_var = self.new_var("graph_out")
        # graph_output_var = None
        cg0 = PyCodegen(tx)
        instructions.append(cg0.create_store(graph_output_var))

        instructions.append(cg0.create_load(graph_output_var))

        self.add_output_instructions(instructions)

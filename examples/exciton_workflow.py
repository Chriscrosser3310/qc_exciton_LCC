from algorithms import GeneralizedQueryAlgorithm, QueryCall, QuerySchedule
from backends import QiskitBackendAdapter, ResourceEstimatorAdapter
from block_encoding import BlockEncoding, BlockEncodingMetadata, BlockEncodingQuery
from exciton.builder import ExcitonBuilder
from exciton.model import OrbitalPartition
from exciton.screening import ConstantScreening


class DemoEncoding(BlockEncoding):
    def __init__(self, name: str, alpha: float) -> None:
        self._meta = BlockEncodingMetadata(name=name, alpha=alpha, ancilla_qubits=2)

    def metadata(self) -> BlockEncodingMetadata:
        return self._meta

    def query(self, request: BlockEncodingQuery) -> object:
        return {"op": "query", "name": self._meta.name, "step": request.step, "p": request.parameters}

    def adjoint_query(self, request: BlockEncodingQuery) -> object:
        return {"op": "adjoint", "name": self._meta.name, "step": request.step, "p": request.parameters}


def main() -> None:
    builder = ExcitonBuilder(
        n_orbitals=4,
        partition=OrbitalPartition(occupied=(0, 1), virtual=(2, 3)),
        screening=ConstantScreening(0.25),
    )
    model = builder.build_minimal()

    schedule = QuerySchedule()
    schedule.append(QueryCall(0, DemoEncoding("lcu", 1.5), BlockEncodingQuery(step=0, parameters={"theta": 0.2})))
    schedule.append(QueryCall(1, DemoEncoding("sparse", 2.0), BlockEncodingQuery(step=1, parameters={"theta": -0.1})))

    algorithm = GeneralizedQueryAlgorithm()
    result = algorithm.run(schedule)

    backend = QiskitBackendAdapter()
    program = backend.compile_operations(result.operations)

    estimator = ResourceEstimatorAdapter(target="logical")
    summary = estimator.export(program)

    print("n_terms", len(model.terms))
    print("program", program.payload)
    print("estimate", summary["summary"])


if __name__ == "__main__":
    main()

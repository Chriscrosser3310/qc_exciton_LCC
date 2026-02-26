from qc_exciton_lcc.algorithms import QueryCall, QuerySchedule
from qc_exciton_lcc.block_encoding import BlockEncoding, BlockEncodingMetadata, BlockEncodingQuery


class _Encoding(BlockEncoding):
    def __init__(self, name: str):
        self._meta = BlockEncodingMetadata(name=name, alpha=1.0, ancilla_qubits=1)

    def metadata(self):
        return self._meta

    def query(self, request: BlockEncodingQuery):
        return (self._meta.name, request.step)

    def adjoint_query(self, request: BlockEncodingQuery):
        return (self._meta.name, -request.step)


def test_schedule_allows_heterogeneous_encodings():
    schedule = QuerySchedule()
    schedule.append(QueryCall(0, _Encoding("enc_a"), BlockEncodingQuery(step=0)))
    schedule.append(QueryCall(1, _Encoding("enc_b"), BlockEncodingQuery(step=1)))

    names = [call.encoding.metadata().name for call in schedule]
    assert names == ["enc_a", "enc_b"]
    assert len(schedule) == 2
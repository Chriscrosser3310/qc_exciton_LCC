from qc_exciton_lcc.exciton.screening import ConstantScreening, ScreeningQuery


def test_constant_screening_returns_same_value():
    model = ConstantScreening(value=0.42)
    q1 = ScreeningQuery(0, 1, 2, 3, omega=0.0)
    q2 = ScreeningQuery(1, 0, 3, 2, omega=2.0)

    assert model.matrix_element(q1) == 0.42
    assert model.matrix_element(q2) == 0.42
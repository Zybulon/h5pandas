"""
h5pandas tests.
"""

import pytest
from pandas.tests.extension import base
from h5pandas import HDF5Dtype
from pandas._typing import Dtype


class TestConstructors(base.BaseConstructorsTests):
    # 15 passed
    pass


class TestCasting(base.BaseCastingTests):
    # 12 passed
    pass


class TestDtype(base.BaseDtypeTests):
    # 22 passed

    def test_construct_from_string_wrong_type_raises(self, dtype):
        with pytest.raises(
            TypeError,
            match="expected string or bytes-like object, got 'int'",
        ):
            type(dtype).construct_from_string(0)


class TestGetitem(base.BaseGetitemTests):
    # 35 passed, 2 xfailed

    def test_getitem_invalid(self, data):
        # TODO: box over scalar, [scalar], (scalar,)?

        msg = (
            r"only integers, slices \(`:`\), ellipsis \(`...`\), numpy.newaxis "
            r"\(`None`\) and integer or boolean arrays are valid indices"
        )
        with pytest.raises(IndexError, match=msg):
            data["foo"]
        with pytest.raises(IndexError, match=msg):
            data[2.5]

        ub = len(data)
        msg = "|".join(
            [
                "list index out of range",  # json
                "index out of bounds",  # pyarrow
                "Out of bounds access",  # Sparse
                f"loc must be an integer between -{ub} and {ub}",  # Sparse
                f"index {ub + 1} is out of bounds for axis 0 with size {ub}",
                f"index -{ub + 1} is out of bounds for axis 0 with size {ub}",
                f"Index \\({ub + 1}\\) out of range for \\(0-{ub - 1}\\)",
                f"Index \\(-1\\) out of range for \\(0-{ub - 1}\\)",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            data[ub + 1]
        with pytest.raises(IndexError, match=msg):
            data[-ub - 1]

    def test_take_pandas_style_negative_raises(self, data, na_value):
        pass


class TestGroupby(base.BaseGroupbyTests):
    # 12 passed
    pass


class TestInterface(base.BaseInterfaceTests):
    #  16 passed
    pass


class TestParsing(base.BaseParsingTests):
    # 2 passed
    # Failed the second time we run it
    # I don't understand the bug yet, two instances of
    # HDF5Dtype seems to have different types
    pass


class TestMethods(base.BaseMethodsTests):
    # 134 passed

    _combine_le_expected_dtype: Dtype = HDF5Dtype("bool")
    pass


class TestMissing(base.BaseMissingTests):
    # 25 passed
    pass


# class TestArithmeticOps(base.BaseArithmeticOpsTests):
#     series_scalar_exc: type[Exception] | None = None
#     frame_scalar_exc: type[Exception] | None = None
#     series_array_exc: type[Exception] | None = None
#     divmod_exc: type[Exception] | None = None

#     # 36 failed, 48 passed
#     pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    # 12 passed
    pass


class TestOpsUtil(base.BaseOpsUtil):
    # Nothing
    pass


class TestUnaryOps(base.BaseUnaryOpsTests):
    # 4 passed
    pass


class TestPrinting(base.BasePrintingTests):
    # 6 passed
    pass


class TestReduce(base.BaseReduceTests):
    #  50 passed, 2 skipped
    def _supports_reduction(self, obj, op_name: str) -> bool:
        # Specify if we expect this reduction to succeed.
        return True

    pass


class TestReshaping(base.BaseReshapingTests):
    # 32 passed
    pass


if __name__ == "__main__":
    retcode = pytest.main(["test_extension.py"])
    retcode = pytest.main(["test_HDFExtensionArray.py"])

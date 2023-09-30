"""
h5pandas tests.
"""

from pandas.tests.extension import base
import pytest


# class TestConstructors(base.BaseConstructorsTests):
#     pass


# class TestCasting(base.BaseCastingTests):
#     pass


# class TestDtype(base.BaseDtypeTests):
#     def test_construct_from_string_wrong_type_raises(self, dtype):
#         with pytest.raises(
#             TypeError,
#             match="expected string or bytes-like object, got 'int'",
#         ):
#             type(dtype).construct_from_string(0)


# class TestGetitem(base.BaseGetitemTests):
#     def test_getitem_invalid(self, data):
#         # TODO: box over scalar, [scalar], (scalar,)?

#         msg = (
#             r"only integers, slices \(`:`\), ellipsis \(`...`\), numpy.newaxis "
#             r"\(`None`\) and integer or boolean arrays are valid indices"
#         )
#         with pytest.raises(IndexError, match=msg):
#             data["foo"]
#         with pytest.raises(IndexError, match=msg):
#             data[2.5]

#         ub = len(data)
#         msg = "|".join(
#             [
#                 "list index out of range",  # json
#                 "index out of bounds",  # pyarrow
#                 "Out of bounds access",  # Sparse
#                 f"loc must be an integer between -{ub} and {ub}",  # Sparse
#                 f"index {ub+1} is out of bounds for axis 0 with size {ub}",
#                 f"index -{ub+1} is out of bounds for axis 0 with size {ub}",
#                 f"Index \({ub+1}\) out of range for \(0-{ub-1}\)",
#                 f"Index \(-1\) out of range for \(0-{ub-1}\)"
#             ]
#         )
#         with pytest.raises(IndexError, match=msg):
#             data[ub + 1]
#         with pytest.raises(IndexError, match=msg):
#             data[-ub - 1]


# class TestGroupby(base.BaseGroupbyTests):
#     pass


# class TestInterface(base.BaseInterfaceTests):
#     pass


# class TestParsing(base.BaseParsingTests):
#     pass


# class TestMethods(base.BaseMethodsTests):
#     pass


# class TestMissing(base.BaseMissingTests):
#     pass


# class TestArithmeticOps(base.BaseArithmeticOpsTests):
#     divmod_exc = None
#     series_scalar_exc = None
#     frame_scalar_exc = None
#     series_array_exc = None
#     pass


# class TestComparisonOps(base.BaseComparisonOpsTests):
#     pass


# class TestOpsUtil(base.BaseOpsUtil):
#     pass


# class TestUnaryOps(base.BaseUnaryOpsTests):
#     pass


# class TestPrinting(base.BasePrintingTests):
#     pass


# class TestBooleanReduce(base.BaseBooleanReduceTests):
#     pass


# class TestNoReduce(base.BaseNoReduceTests):
#     pass


# class TestNumericReduce(base.BaseNumericReduceTests):
#     pass


# class TestReshaping(base.BaseReshapingTests):
#     pass


if __name__ == '__main__':
    # retcode = pytest.main(["test_extension.py"])
    retcode = pytest.main(["test_HDFExtensionArray.py"])

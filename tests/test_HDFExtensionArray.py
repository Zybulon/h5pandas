"""
h5pandas tests.
"""

from pandas.tests.extension import base
import pytest


# class TestConstructors(base.BaseConstructorsTests):
#     # 15 passed/15
#     pass


# class TestCasting(base.BaseCastingTests):
#     # 7 failed, 4 passed, 1 skipped
#     pass


# class TestDtype(base.BaseDtypeTests):
#     # 22 passed
#     def test_construct_from_string_wrong_type_raises(self, dtype):
#         with pytest.raises(
#             TypeError,
#             match="expected string or bytes-like object, got 'int'",
#         ):
#             type(dtype).construct_from_string(0)


# class TestGetitem(base.BaseGetitemTests):
#     # 35 passed, 2 xfailed
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

#     # 12 passed
#     pass


# class TestInterface(base.BaseInterfaceTests):
#     #   1 failed, 13 passed
#     pass


# class TestParsing(base.BaseParsingTests):
#     # 2 failed : should be okay
#     pass


# class TestMethods(base.BaseMethodsTests):
#     # 37 failed, 71 passed, 1 skipped, 4 errors
#     pass


# class TestMissing(base.BaseMissingTests):
#     # 16 passed
#     pass


# class TestArithmeticOps(base.BaseArithmeticOpsTests):

#     series_scalar_exc: type[Exception] | None = None
#     frame_scalar_exc: type[Exception] | None = None
#     series_array_exc: type[Exception] | None = None
#     divmod_exc: type[Exception] | None = None
# divmod_exc = None
# series_scalar_exc = None
# frame_scalar_exc = None
# series_array_exc = None
# 5 failed, 79 passed
# pass


# class TestComparisonOps(base.BaseComparisonOpsTests):
#     # 12 passed
#     pass


# class TestOpsUtil(base.BaseOpsUtil):
#     # Nothing
#     pass


# class TestUnaryOps(base.BaseUnaryOpsTests):
#     # 1 failed, 3 passed,
#     pass


# class TestPrinting(base.BasePrintingTests):
#     # 6 passed
#     pass


# class TestBooleanReduce(base.BaseBooleanReduceTests):
#     # 4 passed, 48 skipped
#     pass


# class TestNoReduce(base.BaseNoReduceTests):
#     # 26 failed, 20 passed, 6 skipped, 16 warnings
#     def _supports_reduction(self, obj, op_name: str) -> bool:
#         # Specify if we expect this reduction to succeed.
# return True
# pass


class TestNumericReduce(base.BaseNumericReduceTests):
    # 26 failed, 16 passed, 10 skipped, 16 warnings
    pass


# class TestReshaping(base.BaseReshapingTests):
#     # 15 failed, 17 passed
#     pass


if __name__ == '__main__':
    # retcode = pytest.main(["test_extension.py"])
    retcode = pytest.main(["test_HDFExtensionArray.py"])

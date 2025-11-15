"""
h5pandas tests.
"""

import pytest
from pandas.tests.extension import base


# class TestConstructors(base.BaseConstructorsTests):
#     # 15 passed/15
#     pass


# class TestCasting(base.BaseCastingTests):
#     # 2 failed, 24 passed, 1 skipped
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
#                 f"index {ub + 1} is out of bounds for axis 0 with size {ub}",
#                 f"index -{ub + 1} is out of bounds for axis 0 with size {ub}",
#                 f"Index \({ub + 1}\) out of range for \(0-{ub - 1}\)",
#                 f"Index \(-1\) out of range for \(0-{ub - 1}\)",
#             ]
#         )
#         with pytest.raises(IndexError, match=msg):
#             data[ub + 1]
#         with pytest.raises(IndexError, match=msg):
#             data[-ub - 1]

#     def test_take_pandas_style_negative_raises(self, data, na_value):
#         pass


# class TestGroupby(base.BaseGroupbyTests):
#     # 12 passed
#     pass


# class TestInterface(base.BaseInterfaceTests):
#     #  14 passed
#     pass


# class TestParsing(base.BaseParsingTests):
#     # 2 failed : should be okay
#     pass


# class TestMethods(base.BaseMethodsTests):
#     # 112 passed, 4 errors

#     @pytest.mark.parametrize("method", ["argmax", "argmin"])
#     def test_argmin_argmax_all_na(self, method, data, na_value):
#         # all missing with skipna=True is the same as empty
#         err_msg = "All-NaN slice encountered"
#         data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
#         with pytest.raises(ValueError, match=err_msg):
#             getattr(data_na, method)()

#     def test_combine_le(self):
#         pass


# class TestMissing(base.BaseMissingTests):
#     # 24 passed
#     pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    series_scalar_exc: type[Exception] | None = None
    frame_scalar_exc: type[Exception] | None = None
    series_array_exc: type[Exception] | None = None
    divmod_exc: type[Exception] | None = None

    # 36 failed, 48 passed
    pass


# class TestComparisonOps(base.BaseComparisonOpsTests):
#     # 12 passed
#     pass


# class TestOpsUtil(base.BaseOpsUtil):
#     # Nothing
#     pass


# class TestUnaryOps(base.BaseUnaryOpsTests):
#     # 4 passed
#     pass


# class TestPrinting(base.BasePrintingTests):
#     # 6 passed
#     pass


# class TestBooleanReduce(base.BaseReduceTests):
#     # 46 passed, 6 skipped
#     def _supports_reduction(self, obj, op_name: str) -> bool:
#         # Specify if we expect this reduction to succeed.
#         return True


# class TestReduce(base.BaseReduceTests):
#     #  46 passed, 6 skipped
#     def _supports_reduction(self, obj, op_name: str) -> bool:
#         # Specify if we expect this reduction to succeed.
#         return True


# class TestReshaping(base.BaseReshapingTests):
# # 32 passed
# pass


if __name__ == "__main__":
    # retcode = pytest.main(["test_extension.py"])
    retcode = pytest.main(["test_HDFExtensionArray.py"])

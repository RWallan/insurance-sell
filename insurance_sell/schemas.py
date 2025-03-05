from functools import partial

import pandera as pa
import pandera.typing as pt

FlagField = partial(pa.Field, ge=0, le=1)


class RawInsuranceSell(pa.DataFrameModel):
    id: pt.Series[int] = pa.Field(unique=True, coerce=True)
    Gender: pt.Series[str] = pa.Field(nullable=True)
    Age: pt.Series[float] = pa.Field(nullable=True)
    HasDrivingLicense: pt.Series[float] = FlagField(nullable=True)
    RegionID: pt.Series[float] = pa.Field(nullable=True)
    Switch: pt.Series[float] = FlagField(nullable=True)
    VehicleAge: pt.Series[str] = pa.Field(nullable=True)
    PastAccident: pt.Series[str] = pa.Field(nullable=True)
    AnnualPremium: pt.Series[str]
    SalesChannelID: pt.Series[int] = pa.Field(coerce=True)
    DaysSinceCreated: pt.Series[int] = pa.Field(coerce=True)
    Result: pt.Series[int] = pa.Field(coerce=True)

    @classmethod
    def get_dtypes(cls):
        """Get the DataFrame Model dtypes.

        Returns:
            A dict with column as key and the dtype as value.
        """
        return {
            col: str(dtype) for col, dtype in cls.to_schema().dtypes.items()
        }

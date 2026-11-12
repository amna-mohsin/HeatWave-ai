"""Pydantic request/response models = input validation (wrong type / missing / out-of-range -> HTTP 422)."""
from datetime import date as Date
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class WeatherInput(BaseModel):
    # required
    temperature_2m_max: float = Field(..., ge=-10, le=55, description="Daily max temperature (C)")
    temperature_2m_min: float = Field(..., ge=-20, le=45, description="Daily min temperature (C)")
    relative_humidity_2m_mean: float = Field(..., ge=0, le=100, description="Mean relative humidity (%)")
    dew_point_2m_mean: float = Field(..., ge=-30, le=35, description="Mean dew point (C)")
    # optional (missing values are imputed by the saved pipeline)
    temperature_2m_mean: Optional[float] = Field(None, ge=-15, le=50)
    apparent_temperature_max: Optional[float] = Field(None, ge=-20, le=70)
    apparent_temperature_mean: Optional[float] = Field(None, ge=-20, le=65)
    precipitation_sum: Optional[float] = Field(None, ge=0, le=600)
    wind_speed_10m_max: Optional[float] = Field(None, ge=0, le=200)
    wind_gusts_10m_max: Optional[float] = Field(None, ge=0, le=300)
    shortwave_radiation_sum: Optional[float] = Field(None, ge=0, le=40)
    et0_fao_evapotranspiration: Optional[float] = Field(None, ge=0, le=20)
    surface_pressure_mean: Optional[float] = Field(None, ge=850, le=1100)
    cloud_cover_mean: Optional[float] = Field(None, ge=0, le=100)
    date: Optional[Date] = Field(None, description="Issue date (defaults to today)")

    @model_validator(mode="after")
    def tmin_not_above_tmax(self):
        if self.temperature_2m_min > self.temperature_2m_max:
            raise ValueError("temperature_2m_min cannot be greater than temperature_2m_max")
        return self


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    risk_band: str
    model: str
    date: str
    source: str

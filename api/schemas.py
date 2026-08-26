from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    amount: float = Field(gt=0)

    account_age: int = Field(ge=0)

    transaction_count: int = Field(ge=0)


class PredictionResponse(BaseModel):

    prediction: int

    fraud_probability: float
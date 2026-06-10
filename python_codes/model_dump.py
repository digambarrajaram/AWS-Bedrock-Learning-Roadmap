from pydantic import BaseModel, ConfigDict # type: ignore
from typing import List
from datetime import datetime

class Address(BaseModel):
    street: str
    city: str
    postal_code: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]
    created_at: datetime
    is_active: bool = True

    model_config = ConfigDict (
        json_encoders =  {
            datetime: lambda v: v.strftime("%d-%m-%Y %H:%M:%S") #dd-mm-yyyy H:M:S
        }
    )       

user = Person(
    name = "Dig",
    age = 20,
    addresses = [
        Address(
            street = "123 Main St",
            city = "Anytown",
            postal_code = "12345"
        )
    ],
    created_at = datetime.now(),
    is_active = True
)

print(user)
print("==============================================\n")
user_dict = user.model_dump()
print(user_dict)
print("==============================================\n")
user_json = user.model_dump_json()
print(user_json)
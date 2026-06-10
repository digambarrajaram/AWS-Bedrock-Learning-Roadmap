from pydantic import BaseModel 
from typing import Optional, List, Dict
#-------------------------------------------
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import computed_field

class Employee(BaseModel):
    id: int
    name: str
    dept: Optional[str]
    skills: List[str]
    info: Dict[str, str]

employee_data = {
    'id' : 1234,
    'name' : 'John Doe',
    'dept' : 'IT',
    'skills' : ['Python', 'Java'],
    'info' : {'email' : 'john.doe@example.com', 'phone' : '123-456-7890'}
}

emp = Employee(**employee_data)
print(emp)


#--------------------------------------------------------------------------------------------

class Student(BaseModel):
    name: str = Field(
        ...,
        min_length=3,
        max_length=50
    )
    age: int = Field(
        ...,
        ge=18,
        le=30,
        description = "Age must be between 18 and 30"
    )
    standard: int = Field(
        ...,
        gt=0,
        le=12,
        description = "Standard must be between 1 and 12"
    )


std = Student(name="John Doe", age=20, standard=11)
print(std)


#----------------------------------------------------------------------------------


#field validator
class Booking(BaseModel):
    user_id: int
    room_id: int
    nights: int = Field(
        ge=1
    )
    rate_per_night: float
    
    @field_validator('user_id')
    def check_user_id(cls, v):
        if v < 1:
            raise ValueError('user_id must be greater than 0')
        return v

booking = Booking(user_id=1, room_id=1, nights=3, rate_per_night=100)
print(booking)



class signUp(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode='before')
    def passwords_match(cls, values):
        if values['password'] != values['confirm_password']:
            raise ValueError('passwords do not match')
        return values


sign_up = signUp(password="secret123", confirm_password="secret123")
print(sign_up)


class square(BaseModel):

    b: int = Field(..., gt=0, le=10)
    h: int = Field(..., gt=0, le=10)

    @computed_field
    @property
    def area_of_square(self) -> int:
        return self.b * self.h

square = square(b=3, h=4)
print(square.area_of_square)
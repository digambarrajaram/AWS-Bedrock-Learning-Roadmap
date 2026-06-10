from pydantic import BaseModel 
from typing import Optional, List


class Lessons(BaseModel):
    lesson_id: int
    name: str

class Chapter(BaseModel):
    chapter_id: int
    name: str
    lessons: List[Lessons]

class Course(BaseModel):
    course_id: int
    name: str
    chapters: List[Chapter]

Course.update_forward_refs()

course = Course(
    course_id=1,
    name="Python Programming",
    chapters=[
        Chapter(
            chapter_id=1,
            name="Chapter 1",
            lessons=[
                Lessons(lesson_id=1, name="Lesson 1"),
                Lessons(lesson_id=2, name="Lesson 2"),
            ],
        ),
        Chapter(
            chapter_id=2,
            name="Chapter 2",
            lessons=[
                Lessons(lesson_id=3, name="Lesson 3"),
                Lessons(lesson_id=4, name="Lesson 4"),
            ],
        ),
    ],
)

print(course.json())
print('-----------------------------------')
print(course.chapters)
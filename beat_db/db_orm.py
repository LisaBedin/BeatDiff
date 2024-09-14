from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey, Float


Base = declarative_base()


class Record(Base):
    __tablename__ = 'records'
    id = Column(Integer, primary_key=True)
    sex = Column(String)
    age = Column(Float)
    dataset_id = Column(String)
    dataset_name = Column(String)
    n_beats = Column(Integer)
    partition_attribution = Column(String)
    target_classes = Column(String)

    def __repr__(self):
        return f"Record({self.id}, {self.dataset_name}, {self.partition_attribution}, {self.target_classes})"


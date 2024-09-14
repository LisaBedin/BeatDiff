import sqlalchemy
from sqlalchemy import *
import numpy as np
from sqlalchemy.orm import declarative_base,relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = 'patient'
    id = Column(Integer, primary_key=True,)
    first_name = Column(String(30))
    last_name = Column(String)
    dob = Column(String)
    gender = Column(String)
    # addresses = relationship("Address", back_populates="patient")
    recording = relationship("Recording", back_populates="patient")

    def __repr__(self):
        # return f"Patient(name={self.first_name!r}, fullname={self.last_name!r})"
        BS_list = []
        for rec in self.recording:
            BS_list = np.append(BS_list,rec.bs)
        return f"Patient(BS={', '.join(BS_list)!r}, first name={self.first_name!r}, last name={self.last_name!r})"


class Pathologies(Base):
    __tablename__ = 'pathologies'
    id = Column(Integer, primary_key=True)
    pathology_class = Column(String)
    pathology_subclass = Column(String)
    VT = Column(Integer)
    recording = relationship("Recording", back_populates="pathology")

    def __repr__(self):
        return f"Pathologies(id={self.id!r}, class={self.pathology_class!r}, subclass={self.pathology_subclass!r}, VT={self.VT!r})"#, record = {self.recording!r})"


class Recording(Base):
    __tablename__ = 'recording'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    bs = Column(String, nullable=False)
    date_recording = Column(String)
    comments = Column(String)
    features = Column(JSON)
    rythm = Column(String)
    bdf_path = Column(String)
    include = Column(Integer)
    processed = Column(Integer)
    patient_id = Column(Integer, ForeignKey('patient.id', onupdate="CASCADE", ondelete="CASCADE"))
    patient = relationship("Patient", back_populates="recording")
    pathology_id = Column(Integer, ForeignKey('pathologies.id', onupdate="CASCADE", ondelete="CASCADE"))
    pathology = relationship("Pathologies", back_populates="recording")


    def __repr__(self):
        # return f"Recording(id={self.id!r}, patient={self.patient.first_name + ' '+self.patient.last_name !r})"#, patho = {self.pathology.pathology_class!r})"
        # return f"Recording(BS={self.bs!r}, patient={self.patient.first_name + ' '+self.patient.last_name !r})"#, patho = {self.pathology.pathology_class!r})"
        return f"Recording(BS={self.bs!r})"

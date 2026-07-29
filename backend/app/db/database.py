import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BACKEND_DIR / "disinfo_system.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)
    f1_score_before = Column(Float)
    f1_score_after = Column(Float)
    status = Column(String)  # "running", "ready_to_promote", "deployed", "failed"
    adapter_path = Column(String)


try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    import logging

    logging.getLogger(__name__).critical("Failed to initialize database at %s: %s", DATABASE_PATH, e)
    raise

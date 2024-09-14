from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import sys

if __name__ == '__main__':
    db_path = sys.argv[1]
    engine = create_engine(f'sqlite:///{db_path}/database.db')

    with engine.connect() as conn:
        ids = conn.execute(text(
            "select target_classes, count(*) from records where partition_attribution like 'Training%' group by target_classes")).fetchall()
        for i in sorted(ids, key=lambda x: x[-1]):
            print(i)


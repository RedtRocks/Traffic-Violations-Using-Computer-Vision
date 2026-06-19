"""
Analytics aggregation queries.
Returns structured data ready for the dashboard.
"""

from sqlalchemy.engine import Engine
from src.database.repository import count_by_type, count_by_date, query_violations


def violation_summary(engine: Engine) -> dict:
    by_type = count_by_type(engine)
    by_date = count_by_date(engine)
    total = sum(by_type.values())
    return {
        "total": total,
        "by_type": by_type,
        "by_date": by_date,
    }


def recent_violations(engine: Engine, limit: int = 50) -> list[dict]:
    return query_violations(engine, limit=limit)


def search(
    engine: Engine,
    violation_type: str | None = None,
    plate_number: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
) -> list[dict]:
    return query_violations(
        engine,
        violation_type=violation_type,
        plate_number=plate_number,
        date_from=date_from,
        date_to=date_to,
        status=status,
        limit=1000,
    )

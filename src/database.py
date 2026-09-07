from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


class LipstickDatabase:
    def __init__(
        self,
        db_path: str | Path = "data/lipstick_recommender.db",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lipsticks (
                    lipstick_id TEXT PRIMARY KEY,
                    brand TEXT,
                    product_name TEXT,
                    shade_name TEXT,
                    finish TEXT,
                    color_family TEXT,

                    L REAL NOT NULL,
                    a REAL NOT NULL,
                    b REAL NOT NULL,
                    chroma REAL NOT NULL,

                    image_path TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    active INTEGER NOT NULL
                        DEFAULT 1
                        CHECK (active IN (0, 1))
                );


                CREATE TABLE IF NOT EXISTS recommendation_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    outfit_image_path TEXT
                );


                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id INTEGER NOT NULL,
                    lipstick_id TEXT NOT NULL,

                    recommendation_order INTEGER NOT NULL,
                    predicted_rank INTEGER NOT NULL,
                    predicted_utility REAL NOT NULL,

                    selected INTEGER NOT NULL
                        DEFAULT 0
                        CHECK (selected IN (0, 1)),

                    feedback INTEGER
                        CHECK (
                            feedback IS NULL
                            OR feedback IN (-1, 0, 1)
                        ),

                    FOREIGN KEY (session_id)
                        REFERENCES recommendation_sessions(session_id)
                        ON DELETE CASCADE,

                    FOREIGN KEY (lipstick_id)
                        REFERENCES lipsticks(lipstick_id)
                );


                CREATE INDEX IF NOT EXISTS
                    idx_recommendations_session
                ON recommendations(session_id);


                CREATE INDEX IF NOT EXISTS
                    idx_recommendations_lipstick
                ON recommendations(lipstick_id);
                """
            )

    # ========================================================
    # LIPSTICK COLLECTION
    # ========================================================

    def add_lipstick(
        self,
        lipstick_id: str,
        L: float,
        a: float,
        b: float,
        chroma: float,
        brand: Optional[str] = None,
        product_name: Optional[str] = None,
        shade_name: Optional[str] = None,
        finish: Optional[str] = None,
        color_family: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO lipsticks (
                    lipstick_id,
                    brand,
                    product_name,
                    shade_name,
                    finish,
                    color_family,
                    L,
                    a,
                    b,
                    chroma,
                    image_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lipstick_id,
                    brand,
                    product_name,
                    shade_name,
                    finish,
                    color_family,
                    float(L),
                    float(a),
                    float(b),
                    float(chroma),
                    image_path,
                ),
            )

    def update_lipstick(
        self,
        lipstick_id: str,
        **fields,
    ) -> None:

        allowed = {
            "brand",
            "product_name",
            "shade_name",
            "finish",
            "color_family",
            "L",
            "a",
            "b",
            "chroma",
            "image_path",
            "active",
        }

        updates = []
        values = []

        for column, value in fields.items():

            if column not in allowed:
                raise ValueError(
                    f"Cannot update field '{column}'."
                )

            if value is None:
                continue

            if column == "active":
                value = int(bool(value))

            updates.append(f"{column} = ?")
            values.append(value)

        if not updates:
            return

        values.append(lipstick_id)

        query = f"""
            UPDATE lipsticks
            SET {", ".join(updates)}
            WHERE lipstick_id = ?
        """

        with self._connect() as conn:
            cursor = conn.execute(
                query,
                values,
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Lipstick '{lipstick_id}' "
                    "does not exist."
                )

    def deactivate_lipstick(
        self,
        lipstick_id: str,
    ) -> None:

        self.update_lipstick(
            lipstick_id,
            active=False,
        )

    def activate_lipstick(
        self,
        lipstick_id: str,
    ) -> None:

        self.update_lipstick(
            lipstick_id,
            active=True,
        )

    def list_lipsticks(
        self,
        active_only: bool = True,
    ) -> pd.DataFrame:

        query = """
            SELECT *
            FROM lipsticks
        """

        if active_only:
            query += " WHERE active = 1"

        query += """
            ORDER BY
                brand,
                product_name,
                shade_name,
                lipstick_id
        """

        with self._connect() as conn:
            return pd.read_sql_query(
                query,
                conn,
            )

    def get_lipstick(
        self,
        lipstick_id: str,
    ) -> Optional[dict]:

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM lipsticks
                WHERE lipstick_id = ?
                """,
                (lipstick_id,),
            ).fetchone()

        return dict(row) if row else None

    # ========================================================
    # SEED PRODUCTION CATALOG
    # ========================================================

    def seed_from_catalog(
        self,
        catalog: pd.DataFrame,
    ) -> int:

        catalog = catalog.copy()

        catalog = catalog.rename(
            columns={
                "L_star": "L",
                "a_star": "a",
                "b_star": "b",
            }
        )

        required = {
            "lipstick_id",
            "L",
            "a",
            "b",
            "chroma",
        }

        missing = required - set(
            catalog.columns
        )

        if missing:
            raise ValueError(
                "Catalog missing required columns: "
                f"{sorted(missing)}"
            )

        optional_columns = [
            "brand",
            "product_name",
            "shade_name",
            "finish",
            "color_family",
            "image_path",
        ]

        inserted = 0

        with self._connect() as conn:

            for _, row in catalog.iterrows():

                metadata = {}

                for column in optional_columns:

                    if (
                        column in catalog.columns
                        and pd.notna(row[column])
                    ):
                        metadata[column] = str(
                            row[column]
                        )
                    else:
                        metadata[column] = None

                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO lipsticks (
                        lipstick_id,
                        brand,
                        product_name,
                        shade_name,
                        finish,
                        color_family,
                        L,
                        a,
                        b,
                        chroma,
                        image_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["lipstick_id"],
                        metadata["brand"],
                        metadata["product_name"],
                        metadata["shade_name"],
                        metadata["finish"],
                        metadata["color_family"],
                        float(row["L"]),
                        float(row["a"]),
                        float(row["b"]),
                        float(row["chroma"]),
                        metadata["image_path"],
                    ),
                )

                inserted += cursor.rowcount

        return inserted

    def seed_from_csv(
        self,
        catalog_path: str | Path,
    ) -> int:

        catalog = pd.read_csv(
            catalog_path
        )

        return self.seed_from_catalog(
            catalog
        )

    # ========================================================
    # RECOMMENDATION SESSIONS
    # ========================================================

    def create_recommendation_session(
        self,
        outfit_image_path: Optional[str] = None,
    ) -> int:

        with self._connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO recommendation_sessions (
                    outfit_image_path
                )
                VALUES (?)
                """,
                (outfit_image_path,),
            )

            return int(
                cursor.lastrowid
            )

    def save_recommendations(
        self,
        session_id: int,
        recommendations: pd.DataFrame,
    ) -> None:

        required = {
            "lipstick_id",
            "predicted_rank",
            "predicted_utility",
        }

        missing = required - set(
            recommendations.columns
        )

        if missing:
            raise ValueError(
                "Recommendations missing columns: "
                f"{sorted(missing)}"
            )

        with self._connect() as conn:

            for order, (_, row) in enumerate(
                recommendations.iterrows(),
                start=1,
            ):

                conn.execute(
                    """
                    INSERT INTO recommendations (
                        session_id,
                        lipstick_id,
                        recommendation_order,
                        predicted_rank,
                        predicted_utility
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(session_id),
                        str(row["lipstick_id"]),
                        int(order),
                        int(row["predicted_rank"]),
                        float(
                            row["predicted_utility"]
                        ),
                    ),
                )

    def save_recommendation_session(
        self,
        recommendations: pd.DataFrame,
        outfit_image_path: Optional[str] = None,
    ) -> int:

        session_id = (
            self.create_recommendation_session(
                outfit_image_path
            )
        )

        self.save_recommendations(
            session_id,
            recommendations,
        )

        return session_id

    # ========================================================
    # USER CHOICE + FEEDBACK
    # ========================================================

    def mark_selected(
        self,
        session_id: int,
        lipstick_id: str,
    ) -> None:

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE recommendations
                SET selected = 0
                WHERE session_id = ?
                """,
                (session_id,),
            )

            cursor = conn.execute(
                """
                UPDATE recommendations
                SET selected = 1
                WHERE session_id = ?
                  AND lipstick_id = ?
                """,
                (
                    session_id,
                    lipstick_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"{lipstick_id} was not "
                    f"recommended in session "
                    f"{session_id}."
                )

    def save_feedback(
        self,
        session_id: int,
        lipstick_id: str,
        feedback: Optional[int],
    ) -> None:

        if feedback not in {
            -1,
            0,
            1,
            None,
        }:
            raise ValueError(
                "Feedback must be "
                "-1, 0, 1, or None."
            )

        with self._connect() as conn:

            cursor = conn.execute(
                """
                UPDATE recommendations
                SET feedback = ?
                WHERE session_id = ?
                  AND lipstick_id = ?
                """,
                (
                    feedback,
                    session_id,
                    lipstick_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"{lipstick_id} was not "
                    f"recommended in session "
                    f"{session_id}."
                )

    # ========================================================
    # HISTORY
    # ========================================================

    def get_session(
        self,
        session_id: int,
    ) -> pd.DataFrame:

        query = """
            SELECT
                s.session_id,
                s.created_at,
                s.outfit_image_path,

                r.recommendation_order,
                r.predicted_rank,
                r.predicted_utility,
                r.selected,
                r.feedback,

                l.lipstick_id,
                l.brand,
                l.product_name,
                l.shade_name,
                l.finish,
                l.color_family,
                l.L,
                l.a,
                l.b,
                l.chroma,
                l.image_path

            FROM recommendation_sessions AS s

            JOIN recommendations AS r
                ON s.session_id =
                   r.session_id

            JOIN lipsticks AS l
                ON r.lipstick_id =
                   l.lipstick_id

            WHERE s.session_id = ?

            ORDER BY
                r.recommendation_order
        """

        with self._connect() as conn:

            return pd.read_sql_query(
                query,
                conn,
                params=(session_id,),
            )

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:

        query = """
            SELECT
                s.session_id,
                s.created_at,
                s.outfit_image_path,

                r.recommendation_order,
                r.predicted_rank,
                r.predicted_utility,
                r.selected,
                r.feedback,

                l.lipstick_id,
                l.brand,
                l.product_name,
                l.shade_name,
                l.finish,
                l.color_family

            FROM recommendation_sessions AS s

            JOIN recommendations AS r
                ON s.session_id =
                   r.session_id

            JOIN lipsticks AS l
                ON r.lipstick_id =
                   l.lipstick_id

            ORDER BY
                s.created_at DESC,
                s.session_id DESC,
                r.recommendation_order ASC
        """

        with self._connect() as conn:

            history = pd.read_sql_query(
                query,
                conn,
            )

        if (
            limit is not None
            and not history.empty
        ):

            latest_sessions = (
                history["session_id"]
                .drop_duplicates()
                .head(limit)
            )

            history = history[
                history["session_id"].isin(
                    latest_sessions
                )
            ].reset_index(drop=True)

        return history
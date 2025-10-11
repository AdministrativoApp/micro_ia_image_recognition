import os
import uuid
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DATABASE = os.getenv("DB_DATABASE", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

def _db():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def ensure_tables():
    with _db() as conn, conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        cur.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm";')
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            sku     TEXT NOT NULL UNIQUE,
            canon   JSONB NOT NULL,            -- mandatory canonical structure
            tokens  TEXT[] NOT NULL,           -- search tokens
            pct     DOUBLE PRECISION,          -- concentration_pct
            vol_ml  DOUBLE PRECISION,          -- volume_ml
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        cur.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
        DROP TRIGGER IF EXISTS trg_products_updated_at ON products;
        CREATE TRIGGER trg_products_updated_at
        BEFORE UPDATE ON products
        FOR EACH ROW EXECUTE PROCEDURE set_updated_at();
        """)
        # Indexes for pruning
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_products_tokens_gin ON products USING GIN (tokens);""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_products_pct ON products (pct);""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_products_vol ON products (vol_ml);""")
        conn.commit()

ensure_tables()

# ────────────────────────────── CRUD ──────────────────────────────
def upsert_product_struct(
    sku: str,
    canon: Dict[str, Any],
    tokens: List[str],
) -> Dict[str, Any]:
    """
    Upsert canonical product row. Returns stored row.
    """
    if not isinstance(canon, dict):
        raise ValueError("canon must be a JSON object")
    if not isinstance(tokens, list):
        raise ValueError("tokens must be a list")

    payload = json.dumps(canon)
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO products (sku, canon, tokens, pct, vol_ml)
            VALUES (%s, %s::jsonb, %s::text[], %s, %s)
            ON CONFLICT (sku) DO UPDATE SET
              canon  = EXCLUDED.canon,
              tokens = EXCLUDED.tokens,
              pct    = EXCLUDED.pct,
              vol_ml = EXCLUDED.vol_ml
            RETURNING id, sku, canon, tokens, pct, vol_ml, created_at, updated_at;
        """, (
            sku.strip(),
            payload,
            tokens,
            canon.get("concentration_pct"),
            canon.get("volume_ml")
        ))
        row = cur.fetchone()
        conn.commit()
        return row

def get_product(sku: str) -> Optional[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, tokens, pct, vol_ml, created_at, updated_at
            FROM products WHERE sku = %s;
        """, (sku.strip(),))
        return cur.fetchone()

def list_products(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, tokens, pct, vol_ml, created_at, updated_at
            FROM products
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s;
        """, (limit, offset))
        return cur.fetchall()

# ────────────────────────────── Candidate search ──────────────────────────────
def search_candidates_by_tokens(
    tokens: List[str],
    pct: Optional[float],
    vol_ml: Optional[float],
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Prune to likely matches using token overlap and coarse numeric filters.
    """
    tok = [t for t in tokens if t]
    if not tok:
        tok = ["__empty__"]  # will likely return none

    # tolerances
    pct_low = pct - 0.1 if pct is not None else None
    pct_hi  = pct + 0.1 if pct is not None else None
    vol_low = vol_ml - 80 if vol_ml is not None else None
    vol_hi  = vol_ml + 80 if vol_ml is not None else None

    where = ["tokens && %s::text[]"]
    params = [tok]

    if pct is not None:
        where.append("pct BETWEEN %s AND %s")
        params.extend([pct_low, pct_hi])
    if vol_ml is not None:
        where.append("vol_ml BETWEEN %s AND %s")
        params.extend([vol_low, vol_hi])

    sql = f"""
        SELECT sku, canon, tokens, pct, vol_ml
        FROM products
        WHERE {" AND ".join(where)}
        ORDER BY updated_at DESC
        LIMIT %s;
    """
    params.append(limit)

    with _db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

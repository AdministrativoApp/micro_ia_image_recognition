# product_store.py
import os
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            sku        TEXT NOT NULL UNIQUE,
            canon      JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        
        # Create updated_at trigger
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
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)
        
        # Indexes on JSONB fields for performance
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_sku ON products (sku);
        CREATE INDEX IF NOT EXISTS idx_canon_name ON products ((canon->>'name'));
        CREATE INDEX IF NOT EXISTS idx_canon_brand ON products ((canon->>'brand'));
        CREATE INDEX IF NOT EXISTS idx_canon_active ON products ((canon->>'active'));
        CREATE INDEX IF NOT EXISTS idx_canon_concentration_pct ON products (((canon->'concentration_pct')::float8));
        CREATE INDEX IF NOT EXISTS idx_canon_volume_ml ON products (((canon->'volume_ml')::float8));
        CREATE INDEX IF NOT EXISTS idx_canon_tablet_count ON products (((canon->'tablet_count')::int));
        """)
        
        conn.commit()

# ────────────────────────────── CRUD ──────────────────────────────
def upsert_product_struct(
    sku: str,
    canon: Dict[str, Any],
    tokens: List[str],  # kept for API compatibility, but not stored
) -> Dict[str, Any]:
    """
    Upsert canonical product row. Returns stored row.
    Note: 'tokens' is not stored in DB (all data lives in 'canon' JSONB).
    """
    if not isinstance(canon, dict):
        raise ValueError("canon must be a JSON object")

    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO products (sku, canon)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (sku) DO UPDATE SET
                canon = EXCLUDED.canon
            RETURNING id, sku, canon, created_at, updated_at;
        """, (
            sku.strip(),
            json.dumps(canon)
        ))
        row = cur.fetchone()
        conn.commit()
        return row

def get_product(sku: str) -> Optional[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, created_at, updated_at
            FROM products WHERE sku = %s;
        """, (sku.strip(),))
        return cur.fetchone()

def list_products(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, created_at, updated_at
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
    tablet_count: Optional[int] = None,
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Search using JSONB fields and token-based filtering in Python.
    Since we don't store 'tokens' in DB, we rely on JSONB indexes + post-filtering.
    """
    with _db() as conn, conn.cursor() as cur:
        # Base query with dynamic filters
        where_clauses = []
        params = []

        # Concentration filter
        if pct is not None:
            where_clauses.append("ABS((canon->>'concentration_pct')::float8 - %s) <= 0.1")
            params.append(pct)

        # Volume filter
        if vol_ml is not None:
            where_clauses.append("ABS((canon->>'volume_ml')::float8 - %s) <= 80")
            params.append(vol_ml)

        # Tablet count filter
        if tablet_count is not None:
            where_clauses.append("ABS((canon->>'tablet_count')::int - %s) <= 5")
            params.append(tablet_count)

        # Build query
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        else:
            where_sql = ""

        sql = f"""
            SELECT sku, canon
            FROM products
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT %s;
        """
        params.append(limit)

        cur.execute(sql, params)
        candidates = cur.fetchall()

        # 🔍 Optional: Add token-based filtering in Python (if needed)
        # For now, rely on numeric pruning + full scoring in main.py
        return candidates

# ────────────────────────────── Other search methods ──────────────────────────────
def search_products_by_text(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT sku, canon
            FROM products
            WHERE 
                canon->>'name' ILIKE %s OR
                canon->>'brand' ILIKE %s OR
                canon->>'active' ILIKE %s
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (
            f"%{query}%",
            f"%{query}%",
            f"%{query}%",
            limit
        ))
        return cur.fetchall()

def get_products_by_manufacturer(manufacturer: str, limit: int = 100) -> List[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT sku, canon
            FROM products
            WHERE canon->>'brand' ILIKE %s
            ORDER BY canon->>'name', updated_at DESC
            LIMIT %s;
        """, (f"%{manufacturer}%", limit))
        return cur.fetchall()

def get_similar_products(sku: str, limit: int = 20) -> List[Dict[str, Any]]:
    product = get_product(sku)
    if not product:
        return []
    
    canon = product['canon']
    return search_candidates_by_tokens(
        tokens=[],  # not used
        pct=canon.get('concentration_pct'),
        vol_ml=canon.get('volume_ml'),
        tablet_count=canon.get('tablet_count'),
        limit=limit
    )

# ────────────────────────────── Database maintenance ──────────────────────────────
def get_database_info() -> Dict[str, Any]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total_products,
                AVG((canon->>'concentration_pct')::float8) as avg_concentration,
                AVG((canon->>'volume_ml')::float8) as avg_volume,
                AVG((canon->>'tablet_count')::int) as avg_tablet_count
            FROM products
            WHERE canon ? 'concentration_pct' OR canon ? 'volume_ml' OR canon ? 'tablet_count';
        """)
        stats = cur.fetchone()
        
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'products'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        
        return {
            "statistics": stats,
            "columns": columns
        }
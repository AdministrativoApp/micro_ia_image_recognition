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
        
        # Check if tablet_count column exists, if not add it
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='products' and column_name='tablet_count';
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE products ADD COLUMN tablet_count INTEGER;")
        
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
        
        # Create tablet_count index if it doesn't exist
        cur.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'products' AND indexname = 'idx_products_tablet_count';
        """)
        if not cur.fetchone():
            cur.execute("""CREATE INDEX idx_products_tablet_count ON products (tablet_count);""")
        
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
            INSERT INTO products (sku, canon, tokens, pct, vol_ml, tablet_count)
            VALUES (%s, %s::jsonb, %s::text[], %s, %s, %s)
            ON CONFLICT (sku) DO UPDATE SET
              canon        = EXCLUDED.canon,
              tokens       = EXCLUDED.tokens,
              pct          = EXCLUDED.pct,
              vol_ml       = EXCLUDED.vol_ml,
              tablet_count = EXCLUDED.tablet_count
            RETURNING id, sku, canon, tokens, pct, vol_ml, tablet_count, created_at, updated_at;
        """, (
            sku.strip(),
            payload,
            tokens,
            canon.get("concentration_pct"),
            canon.get("volume_ml"),
            canon.get("tablet_count")
        ))
        row = cur.fetchone()
        conn.commit()
        return row

def get_product(sku: str) -> Optional[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, tokens, pct, vol_ml, tablet_count, created_at, updated_at
            FROM products WHERE sku = %s;
        """, (sku.strip(),))
        return cur.fetchone()

def list_products(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, canon, tokens, pct, vol_ml, tablet_count, created_at, updated_at
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
    Prune to likely matches using token overlap and coarse numeric filters.
    Enhanced with tablet count support for medication products.
    """
    tok = [t for t in tokens if t]
    if not tok:
        tok = ["__empty__"]  # will likely return none

    # Build WHERE conditions dynamically
    where_conditions = ["tokens && %s::text[]"]
    params = [tok]

    # Concentration filter
    if pct is not None:
        pct_low = pct - 0.1
        pct_hi = pct + 0.1
        where_conditions.append("pct BETWEEN %s AND %s")
        params.extend([pct_low, pct_hi])

    # Volume filter
    if vol_ml is not None:
        vol_low = vol_ml - 80
        vol_hi = vol_ml + 80
        where_conditions.append("vol_ml BETWEEN %s AND %s")
        params.extend([vol_low, vol_hi])

    # Tablet count filter - only apply if column exists and value is provided
    if tablet_count is not None:
        # For tablets, use a tighter tolerance since counts are discrete
        tab_low = tablet_count - 5
        tab_hi = tablet_count + 5
        where_conditions.append("tablet_count BETWEEN %s AND %s")
        params.extend([tab_low, tab_hi])

    # Build the base SQL
    sql = f"""
        SELECT sku, canon, tokens, pct, vol_ml, tablet_count
        FROM products
        WHERE {" AND ".join(where_conditions)}
    """

    # Add ordering - handle potential NULL values for tablet_count
    sql += """
        ORDER BY 
            -- Prioritize exact matches on key numeric fields
            CASE WHEN pct = %s THEN 0 ELSE 1 END,
            CASE WHEN vol_ml = %s THEN 0 ELSE 1 END,
            updated_at DESC
        LIMIT %s;
    """
    
    # Add exact match parameters for ordering
    params.extend([pct, vol_ml, limit])

    with _db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

# ────────────────────────────── Enhanced search methods ──────────────────────────────
def search_products_by_text(
    query: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Search products by text in name, brand, active ingredient, etc.
    Uses PostgreSQL full-text search capabilities.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT sku, canon, tokens, pct, vol_ml, tablet_count
            FROM products
            WHERE 
                tokens && regexp_split_to_array(lower(%s), '[^a-z0-9]+') OR
                canon->>'name' ILIKE %s OR
                canon->>'brand' ILIKE %s OR
                canon->>'active' ILIKE %s
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (
            query.lower(),
            f"%{query}%",
            f"%{query}%", 
            f"%{query}%",
            limit
        ))
        return cur.fetchall()

def get_products_by_manufacturer(
    manufacturer: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get products by manufacturer/brand name.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT sku, canon, tokens, pct, vol_ml, tablet_count
            FROM products
            WHERE canon->>'brand' ILIKE %s
            ORDER BY canon->>'name', updated_at DESC
            LIMIT %s;
        """, (f"%{manufacturer}%", limit))
        return cur.fetchall()

def get_similar_products(
    sku: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find products similar to the given SKU based on tokens and characteristics.
    """
    product = get_product(sku)
    if not product:
        return []
    
    canon = product.get('canon', {})
    tokens = product.get('tokens', [])
    
    return search_candidates_by_tokens(
        tokens=tokens,
        pct=canon.get('concentration_pct'),
        vol_ml=canon.get('volume_ml'),
        tablet_count=canon.get('tablet_count'),
        limit=limit
    )

# ────────────────────────────── Database maintenance ──────────────────────────────
def get_database_info() -> Dict[str, Any]:
    """
    Get database information including column structure.
    """
    with _db() as conn, conn.cursor() as cur:
        # Get table info
        cur.execute("""
            SELECT 
                COUNT(*) as total_products,
                COUNT(tablet_count) as products_with_tablet_count,
                AVG(pct) as avg_concentration,
                AVG(vol_ml) as avg_volume,
                AVG(tablet_count) as avg_tablet_count
            FROM products;
        """)
        stats = cur.fetchone()
        
        # Get column info
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'products'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        
        return {
            "statistics": stats,
            "columns": columns
        }
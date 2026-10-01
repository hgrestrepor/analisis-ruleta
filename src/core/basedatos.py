"""Persistencia global en SQLite.

Guarda cada captura analizada y cada tirada reconocida. Las métricas del
dashboard se derivan siempre de la tabla `tiradas`, de modo que el histórico
sobrevive a reinicios y se puede reprocesar sin volver a leer las imágenes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from core import ruleta_reglas as R

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DB = RAIZ / "data" / "analizador.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS capturas (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    carpeta      TEXT NOT NULL DEFAULT '',
    archivo      TEXT NOT NULL,
    ruta         TEXT NOT NULL,
    analizada_en TEXT NOT NULL,
    modo         TEXT NOT NULL,
    n_detectadas INTEGER NOT NULL,
    n_reconocidas INTEGER NOT NULL,
    n_dudas      INTEGER NOT NULL,
    n_conflicto  INTEGER NOT NULL DEFAULT 0,
    confianza    REAL,
    rejilla      TEXT,
    estado       TEXT NOT NULL DEFAULT 'ok',
    error        TEXT
);

CREATE TABLE IF NOT EXISTS tiradas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    captura_id  INTEGER NOT NULL REFERENCES capturas(id) ON DELETE CASCADE,
    posicion    INTEGER NOT NULL,
    numero      INTEGER NOT NULL,
    confianza   REAL,
    color       TEXT,
    estado      TEXT NOT NULL DEFAULT 'ok',
    UNIQUE (captura_id, posicion)
);

CREATE TABLE IF NOT EXISTS ajustes_rejilla (
    nombre      TEXT PRIMARY KEY,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    paso_x      REAL NOT NULL,
    paso_y      REAL NOT NULL,
    columnas    INTEGER NOT NULL,
    filas       INTEGER NOT NULL,
    actualizado TEXT NOT NULL
);
"""

_candado = threading.Lock()


def conexion() -> sqlite3.Connection:
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RUTA_DB, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con


def inicializar() -> None:
    with _candado, conexion() as con:
        con.executescript(ESQUEMA)
        _migrar_carpeta(con)
        con.commit()


def _migrar_carpeta(con: sqlite3.Connection) -> None:
    """Añade la columna `carpeta` a las bases creadas antes de los casinos.

    Las capturas antigas quedaban en la raíz de `imagenes/capturas`, así que
    se marcan con la raíz (vacío) y el historial no se pierde al migrar.
    """
    columnas = {f["name"] for f in con.execute("PRAGMA table_info(capturas)")}
    if "carpeta" not in columnas:
        con.execute(
            "ALTER TABLE capturas ADD COLUMN carpeta TEXT NOT NULL DEFAULT ''"
        )


def registrar_captura(
    archivo: str,
    ruta: str,
    analizada_en: str,
    modo: str,
    n_detectadas: int,
    n_reconocidas: int,
    n_dudas: int,
    n_conflicto: int,
    confianza: float,
    rejilla: dict | None,
    estado: str = "ok",
    error: str | None = None,
    tiradas: list[tuple[int, float, str, str]] | None = None,
    carpeta: str = "",
) -> int:
    """Guarda una captura y sus tiradas. `tiradas` = (numero, confianza, color, estado)."""
    with _candado, conexion() as con:
        cur = con.execute(
            """INSERT INTO capturas
               (carpeta, archivo, ruta, analizada_en, modo, n_detectadas, n_reconocidas,
                n_dudas, n_conflicto, confianza, rejilla, estado, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                carpeta or "",
                archivo,
                ruta,
                analizada_en,
                modo,
                n_detectadas,
                n_reconocidas,
                n_dudas,
                n_conflicto,
                confianza,
                json.dumps(rejilla) if rejilla else None,
                estado,
                error,
            ),
        )
        cid = int(cur.lastrowid)
        if tiradas:
            con.executemany(
                """INSERT INTO tiradas (captura_id, posicion, numero, confianza, color, estado)
                   VALUES (?,?,?,?,?,?)""",
                [(cid, i, n, c, col, est) for i, (n, c, col, est) in enumerate(tiradas)],
            )
        return cid


def borrar_captura(captura_id: int) -> None:
    with _candado, conexion() as con:
        con.execute("DELETE FROM tiradas WHERE captura_id = ?", (captura_id,))
        con.execute("DELETE FROM capturas WHERE id = ?", (captura_id,))


def borrar_por_archivo(archivo: str, carpeta: str = "") -> int:
    """Elimina los registros previos de ese archivo dentro de su carpeta.

    Reprocesar la misma captura debe *actualizar* sus tiradas, no duplicarlas:
    si no, pulsar «Analizar» dos veces inflaría todas las estadísticas.
    La carpeta importa: `betplay/1.jpg` y `wplay/1.jpg` son archivos
    distintos aunque se llamen igual.
    """
    with _candado, conexion() as con:
        ids = [
            int(f["id"])
            for f in con.execute(
                "SELECT id FROM capturas WHERE archivo = ? AND carpeta = ?",
                (archivo, carpeta or ""),
            )
        ]
        for cid in ids:
            con.execute("DELETE FROM tiradas WHERE captura_id = ?", (cid,))
            con.execute("DELETE FROM capturas WHERE id = ?", (cid,))
        return len(ids)


def limpiar(carpeta: str | None = None) -> None:
    """Borra el historial. Sin `carpeta`, lo borra todo."""
    with _candado, conexion() as con:
        if carpeta is None:
            con.execute("DELETE FROM tiradas")
            con.execute("DELETE FROM capturas")
        else:
            ids = [
                int(f["id"])
                for f in con.execute(
                    "SELECT id FROM capturas WHERE carpeta = ?", (carpeta,)
                )
            ]
            for cid in ids:
                con.execute("DELETE FROM tiradas WHERE captura_id = ?", (cid,))
            con.execute("DELETE FROM capturas WHERE carpeta = ?", (carpeta,))


def todas_las_tiradas(carpeta: str | None = None) -> list[int]:
    """Tiradas en orden cronológico real (captura analizada, luego posición)."""
    with _candado, conexion() as con:
        if carpeta is None:
            filas = con.execute(
                """SELECT t.numero
                   FROM tiradas t JOIN capturas c ON c.id = t.captura_id
                   WHERE t.estado = 'ok' AND c.estado = 'ok'
                   ORDER BY c.analizada_en, c.id, t.posicion"""
            ).fetchall()
        else:
            filas = con.execute(
                """SELECT t.numero
                   FROM tiradas t JOIN capturas c ON c.id = t.captura_id
                   WHERE t.estado = 'ok' AND c.estado = 'ok' AND c.carpeta = ?
                   ORDER BY c.analizada_en, c.id, t.posicion""",
                (carpeta,),
            ).fetchall()
    return [int(f["numero"]) for f in filas]


def podar_capturas_inexistentes(raiz) -> int:
    """Borra los registros cuyo archivo ya no está en el disco.

    Si borras una captura y vuelves a analizar, las filas viejas seguían en la
    base y el contador de imágenes decía más de las que hay. Cada registro se
    busca en *su* carpeta, así que basta con pasar la raíz de `capturas`.
    """
    raiz = Path(raiz)
    with _candado, conexion() as con:
        filas = con.execute("SELECT id, carpeta, archivo FROM capturas").fetchall()
        borrados = 0
        for f in filas:
            destino = raiz / (f["carpeta"] or "") / f["archivo"]
            if destino.exists():
                continue
            con.execute("DELETE FROM tiradas WHERE captura_id = ?", (int(f["id"]),))
            con.execute("DELETE FROM capturas WHERE id = ?", (int(f["id"]),))
            borrados += 1
        con.commit()
    return borrados


def resumen_capturas(carpeta: str | None = None) -> dict:
    """Resumen del historial. Sin `carpeta`, mezcla todos los casinos."""
    filtro, params = ("WHERE carpeta = ?", (carpeta,)) if carpeta is not None else ("", ())
    with _candado, conexion() as con:
        fila = con.execute(
            f"""SELECT COUNT(*) AS capturas,
                       SUM(CASE WHEN estado='ok' THEN 1 ELSE 0 END) AS ok,
                       SUM(CASE WHEN estado='error' THEN 1 ELSE 0 END) AS errores,
                       COALESCE(SUM(n_reconocidas),0) AS tiradas,
                       COALESCE(SUM(n_dudas),0) AS dudas,
                       COALESCE(SUM(n_conflicto),0) AS conflictos
                FROM capturas {filtro}""",
            params,
        ).fetchone()
        ultima = con.execute(
            f"SELECT MAX(analizada_en) AS u FROM capturas {filtro}"
            + (" AND" if filtro else " WHERE") + " estado='ok'",
            params,
        ).fetchone()
    return {
        "capturas": int(fila["capturas"] or 0),
        "capturas_ok": int(fila["ok"] or 0),
        "capturas_error": int(fila["errores"] or 0),
        "tiradas": int(fila["tiradas"] or 0),
        "dudas": int(fila["dudas"] or 0),
        "conflictos": int(fila["conflictos"] or 0),
        "ultima_analisis": ultima["u"],
    }


def ultimas_capturas(limite: int = 50, carpeta: str | None = None) -> list[dict]:
    filtro, params = ("WHERE carpeta = ?", (carpeta,)) if carpeta is not None else ("", ())
    with _candado, conexion() as con:
        filas = con.execute(
            f"""SELECT id, carpeta, archivo, analizada_en, modo, n_detectadas, n_reconocidas,
                       n_dudas, n_conflicto, confianza, estado, error
                FROM capturas {filtro} ORDER BY id DESC LIMIT ?""",
            (*params, limite),
        ).fetchall()
    return [dict(f) for f in filas]


def secuencia_tiradas(limite: int = 500, carpeta: str | None = None) -> list[dict]:
    """Todas las tiradas en orden cronológico, con su imagen de origen.

    Es la lista que permite verificar a mano qué se leyó de cada captura.
    """
    filtro, params = ("AND c.carpeta = ?", (carpeta,)) if carpeta is not None else ("", ())
    with _candado, conexion() as con:
        filas = con.execute(
            f"""SELECT t.posicion, t.numero, t.confianza, t.color, t.estado,
                       c.carpeta, c.archivo, c.modo, c.analizada_en, c.rejilla
                FROM tiradas t JOIN capturas c ON c.id = t.captura_id
                WHERE t.estado = 'ok' AND c.estado = 'ok' {filtro}
                ORDER BY c.analizada_en, c.id, t.posicion
                LIMIT ?""",
            (*params, max(1, limite)),
        ).fetchall()
    salida = []
    for i, f in enumerate(filas, start=1):
        salida.append(
            {
                "pos": i,
                "posicion_en_captura": f["posicion"],
                "numero": f["numero"],
                "docena": R.dozen_de(f["numero"]),
                "color": R.color_de(f["numero"]),
                "confianza": round(f["confianza"] or 0.0, 3),
                "estado": f["estado"],
                "carpeta": f["carpeta"] or "",
                "archivo": f["archivo"],
                "modo": f["modo"],
                "analizada_en": f["analizada_en"],
            }
        )
    return salida


def guardar_ajuste(nombre: str, rejilla: dict) -> None:
    from datetime import datetime, timezone

    with _candado, conexion() as con:
        con.execute(
            """INSERT INTO ajustes_rejilla
               (nombre, x, y, paso_x, paso_y, columnas, filas, actualizado)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(nombre) DO UPDATE SET
                 x=excluded.x, y=excluded.y, paso_x=excluded.paso_x,
                 paso_y=excluded.paso_y, columnas=excluded.columnas,
                 filas=excluded.filas, actualizado=excluded.actualizado""",
            (
                nombre,
                float(rejilla["x"]),
                float(rejilla["y"]),
                float(rejilla["paso_x"]),
                float(rejilla["paso_y"]),
                int(rejilla["columnas"]),
                int(rejilla["filas"]),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )


def listar_ajustes() -> list[dict]:
    with _candado, conexion() as con:
        filas = con.execute(
            "SELECT * FROM ajustes_rejilla ORDER BY nombre"
        ).fetchall()
    return [dict(f) for f in filas]

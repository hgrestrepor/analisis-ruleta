"""Estrategia de escalada al tercer golpe de una misma docena, en pesos colombianos.

Regla simulada
--------------
Cuando una mismaienza a caer dos veces **seguidas**, se apuesta a que en la
tirada siguiente ya no sale: se betean los 24 números de las otras dos docenas
más el cero, 25 números a 2.500 cada uno.

Si esa tirada vuelve a ser de la misma docena —el tercer golpe—, se pierde la
apuesta de 2.500 y se vuelve a apostar a lo mismo, pero al triple: 7.500 por
número. Si vuelve a caer esa misma docena por cuarta vez, se pierde también la
de 7.500 y se regresa a 2.500, y así sucesivamente mientras la racha siga.

    racha de 2   → se apuesta 2.500 por número a los 25 ajenos
    racha de 3   → se apuesta 7.500 por número a los 25 ajenos
    racha de 4   → se vuelve a 2.500
    racha de 5   → se vuelve a 7.500
    ...

El cero no pertenece a ninguna docena, así que rompe la racha; aun así está
apostado, de modo que si sale en el momento de la apuesta esa apuesta gana.

Cálculo de una apuesta (ruleta europea, cero, pago 35:1 en número simple)
    Apostado            = 25 números × 2.500 = 62.500
    Si acierta uno:     gana 35 × 2.500 = 87.500
                        pierde los otros 24 × 2.500 = 60.000
                        neto = +27.500
    Si no acierta:      sale la decidedly repetida (o nada útil)
                        neto = −62.500

    Apostado            = 25 números × 7.500 = 187.500
    Si acierta uno:     neto = +82.500
    Si no acierta:      neto = −187.500

Nota importante: la esperanza matemática es negativa en los dos tramos. Apostar
25 números acierta 25 de cada 37 veces y paga 35:1, así que se recuperan 67,57
por cada 100 jugados. La escalada multiplica tanto las ganancias como las
pérdidas, y como se apuesta el triple justo cuando va a caer la misma docena, el
tercer tramo pierde con frecuencia. El simulador muestra el resultado histórico
real, no una ganancia esperada.
"""

from __future__ import annotations

from core.simulador import PAGO_NUMERO_SIMPLE, _docena_de, cop, numeros_apostados, netos

MONEDA = "COP"


def _monto_para(racha: int, base: int) -> int:
    """Cantidad a apostar por número cuando la racha va por `racha` golpes.

    2.500 en la segunda, 7.500 en la tercera, 2.500 en la cuarta… El triple entra
    en cada golpe impar de la racha.
    """
    return base if racha % 2 == 0 else base * 3


def _movimientos(tiradas: list[int], base: int, incluir_cero: bool) -> list[dict]:
    """Recorre la secuencia y anota cada apuesta con su resultado.

    La apuesta se abre cuando la racha alcanza la segunda repetición y se
    resuelve con la tirada siguiente, sea cual sea. Solo puede haber una abierta
    a la vez, porque una racha se repite de una en una.
    """
    tiradas = [int(t) for t in tiradas]
    docenas = [_docena_de(n) for n in tiradas]
    movimientos: list[dict] = []

    # racha que termina en cada posición: (docena, largo, desde)
    racha_actual: str | None = None
    largo = 0
    desde = 0

    for i, d in enumerate(docenas):
        if d is not None and d == racha_actual:
            largo += 1
        else:
            racha_actual, largo, desde = d, (1 if d is not None else 0), i

        # la apuesta se abre tras el segundo golpe y se resuelve en la siguiente
        if d is None or largo < 2:
            continue
        if i + 1 >= len(tiradas):
            movimientos.append(
                {
                    "apertura_posicion": i,
                    "docena_repetida": d,
                    "racha_al_disparar": largo,
                    "apuesta_unitaria": _monto_para(largo, base),
                    "sin_resolver": True,
                    "resultado": None,
                    "acierto": None,
                    "neto": 0,
                    "apostado": 0,
                    "devuelto": 0,
                }
            )
            break

        unitario = _monto_para(largo, base)
        numeros = numeros_apostados(d, incluir_cero)
        resultado = tiradas[i + 1]
        acierto = resultado in numeros
        gana, pierde = netos(unitario, len(numeros))
        movimientos.append(
            {
                "apertura_posicion": i,
                "resolucion_posicion": i + 1,
                "docena_repetida": d,
                "racha_al_disparar": largo,
                "racha_tras_el_impacto": (largo + 1 if docenas[i + 1] == d else largo),
                "apuesta_unitaria": unitario,
                "cuantos_numeros": len(numeros),
                "apostado": unitario * len(numeros),
                "devuelto": unitario * (1 + PAGO_NUMERO_SIMPLE) if acierto else 0,
                "resultado": resultado,
                "acierto": acierto,
                "neto": gana if acierto else pierde,
            }
        )
    return movimientos


def simular(
    tiradas,
    base: int = 2500,
    incluir_cero: bool = True,
) -> dict:
    """Simula la escalada de 2.500 a 7.500 sobre la racha de docenas."""
    tiradas = [int(t) for t in tiradas]
    base = int(base)
    if base <= 0:
        raise ValueError("La apuesta base debe ser mayor que cero.")

    movimientos = _movimientos(tiradas, base, incluir_cero)
    resueltas = [m for m in movimientos if not m.get("sin_resolver")]
    ganadoras = [m for m in resueltas if m["acierto"]]
    perdedoras = [m for m in resueltas if not m["acierto"]]

    total_apostado = sum(m["apostado"] for m in resueltas)
    total_devuelto = sum(m["devuelto"] for m in resueltas)
    neto = sum(m["neto"] for m in resueltas)
    # separado por tipo de resultado, que es como se lee sin sumar de más
    ganado = sum(m["neto"] for m in resueltas if m["neto"] > 0)
    perdido = -sum(m["neto"] for m in resueltas if m["neto"] < 0)

    base_m = [m for m in resueltas if m["racha_al_disparar"] % 2 == 0]
    triple_m = [m for m in resueltas if m["racha_al_disparar"] % 2 == 1]
    tras_dos = [m for m in resueltas if m["racha_al_disparar"] == 2]
    tras_tres = [m for m in resueltas if m["racha_al_disparar"] == 3]

    cuantos = len(numeros_apostados("1-12", incluir_cero))
    prob = cuantos / 37.0
    ev_base = prob * netos(base, cuantos)[0] + (1 - prob) * netos(base, cuantos)[1]
    ev_triple = prob * netos(base * 3, cuantos)[0] + (1 - prob) * netos(base * 3, cuantos)[1]

    return {
        "moneda": MONEDA,
        "apuesta_base": base,
        "apuesta_triple": base * 3,
        "incluye_cero": incluir_cero,
        "numeros_por_apuesta": cuantos,
        "tiradas": len(tiradas),
        "apuestas": len(resueltas),
        "apuestas_sin_resolver": len(movimientos) - len(resueltas),
        "aciertos": len(ganadoras),
        "fallos": len(perdedoras),
        "tasa_acerto": round(100.0 * len(ganadoras) / len(resueltas), 2) if resueltas else None,
        "total_apostado": total_apostado,
        "total_devuelto": total_devuelto,
        "neto": neto,
        "ganado": ganado,
        "perdido": perdido,
        "roi_pct": round(100.0 * neto / total_apostado, 2) if total_apostado else None,
        "desglose": {
            "a_2_500": _bloque(base_m),
            "a_7_500": _bloque(triple_m),
        },
        "impactos": {
            "tras_dos_golpes": _bloque(tras_dos),
            "tras_tres_golpes": _bloque(tras_tres),
        },
        "mejor": max((m["neto"] for m in resueltas), default=0),
        "peor": min((m["neto"] for m in resueltas), default=0),
        "mejor_apuesta": max(resueltas, key=lambda m: m["neto"], default=None),
        "peor_apuesta": min(resueltas, key=lambda m: m["neto"], default=None),
        "por_docena": _resumen_por_docena(resueltas),
        "movimientos": movimientos,
        "regla": _texto_regla(base, incluir_cero),
        "esperanza_teorica_base": round(ev_base),
        "esperanza_teorica_triple": round(ev_triple),
        "prob_teorica_acierto": round(100 * prob, 2),
        "aviso": (
            f"Aunque acierte el {round(100 * prob, 1)}% de las veces, esta estrategia "
            f"pierde en promedio {cop(abs(ev_base))} COP por apuesta de {cop(base)} y "
            f"{cop(abs(ev_triple))} COP por apuesta de {cop(base * 3)}. La escalada "
            "sube las dos cosas por igual."
        ),
    }


def _bloque(movimientos: list[dict]) -> dict:
    apostado = sum(m["apostado"] for m in movimientos)
    devuelto = sum(m["devuelto"] for m in movimientos)
    return {
        "apuestas": len(movimientos),
        "aciertos": sum(1 for m in movimientos if m["acierto"]),
        "fallos": sum(1 for m in movimientos if not m["acierto"]),
        "total_apostado": apostado,
        "total_devuelto": devuelto,
        "neto": devuelto - apostado,
    }


def _texto_regla(base: int, incluir_cero: bool) -> str:
    otros = (
        "los otros 24 números más el 0 (25 números en total)"
        if incluir_cero
        else "los otros 24 números"
    )
    return (
        f"Cuando una misma docena sale 2 veces seguidas, se betean {otros} a "
        f"{cop(base)} cada uno ({cop(base * len(numeros_apostados('1-12', incluir_cero)))} "
        f"en total). Si vuelve a salir esa misma docena, se pierde y se repite la "
        f"apuesta al triple: {cop(base * 3)} por número "
        f"({cop(base * 3 * len(numeros_apostados('1-12', incluir_cero)))}). "
        f"Si cae una cuarta vez, se pierde también esa y se vuelve a {cop(base)}, "
        "alternando mientras la racha siga."
    )


def _resumen_por_docena(movimientos: list[dict]) -> dict:
    salida: dict = {}
    for m in movimientos:
        d = m["docena_repetida"]
        fila = salida.setdefault(d, {"apuestas": 0, "aciertos": 0, "neto": 0, "apostado": 0})
        fila["apuestas"] += 1
        fila["apostado"] += m["apostado"]
        if m["acierto"]:
            fila["aciertos"] += 1
        fila["neto"] += m["neto"]
    return salida
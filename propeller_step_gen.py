#!/usr/bin/env python3
"""
Генератор 3D-модели воздушного винта в формате STEP.

Читает propeller_params.json, экспортированный из HTML-калькулятора,
и строит параметрическую твердотельную модель: втулка с отверстием под ось
+ лопасти, построенные лофтом по сечениям NACA 4412 с закруткой.

Установка:
    pip install cadquery
    (или conda install -c cadquery -c conda-forge cadquery)

Запуск:
    python propeller_step_gen.py propeller_params.json
    python propeller_step_gen.py propeller_params.json -o vint.step --fillet 2.5
    python propeller_step_gen.py --demo          # без JSON, встроенные параметры

Результат: STEP (AP214) + опционально STL. Дальше — слайсер.
"""

import json
import math
import argparse
import sys
from pathlib import Path

try:
    import cadquery as cq
except ImportError:
    sys.exit("Не найден cadquery. Установите: pip install cadquery")


# ----------------------------------------------------------------------
# Профиль NACA 4412
# ----------------------------------------------------------------------
def naca4412_points(n=70, close=True):
    """Замкнутый контур NACA 4412 в единичной хорде.

    Порядок: задняя кромка -> по верхней поверхности к носку ->
    по нижней поверхности назад к задней кромке.
    Косинусное сгущение точек к носку и хвосту.
    """
    m, p, t = 0.04, 0.40, 0.12
    upper, lower = [], []

    for i in range(n):
        beta = math.pi * i / (n - 1)
        x = (1.0 - math.cos(beta)) / 2.0

        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x
                      - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        if x < p:
            yc = m / p**2 * (2 * p * x - x**2)
            dyc = 2 * m / p**2 * (p - x)
        else:
            yc = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
            dyc = 2 * m / (1 - p)**2 * (p - x)

        th = math.atan(dyc)
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))

    pts = list(reversed(upper)) + lower[1:]
    if close and (abs(pts[0][0] - pts[-1][0]) > 1e-9
                  or abs(pts[0][1] - pts[-1][1]) > 1e-9):
        pts.append(pts[0])
    return pts


FOIL_UNIT = naca4412_points()
FOIL_MAX_T = 0.12  # относительная толщина базового профиля


# ----------------------------------------------------------------------
# Размещение сечения в пространстве
# ----------------------------------------------------------------------
def section_wire(r_mm, chord_mm, beta_deg, thick_mm,
                 pitch_axis=0.30, foil=None):
    """Строит замкнутый Wire одного сечения лопасти.

    Система координат модели:
        Z — ось вращения винта,
        сечение лежит в плоскости, отстоящей от оси на r_mm вдоль X.

    Профиль:
      1) масштабируется по хорде,
      2) вертикально растягивается под требуемую толщину thick_mm,
      3) поворачивается на угол установки beta вокруг оси изменения шага,
      4) переносится на радиус r и укладывается в плоскость (Y — окружное
         направление, Z — осевое).
    """
    foil = foil or FOIL_UNIT
    # масштаб толщины: базовый профиль имеет 12% хорды
    t_scale = (thick_mm / chord_mm) / FOIL_MAX_T if chord_mm > 1e-6 else 1.0

    b = math.radians(beta_deg)
    cb, sb = math.cos(b), math.sin(b)

    pts3d = []
    for xu, yu in foil:
        # в плоскости профиля: xc — вдоль хорды, yc — по нормали
        xc = (xu - pitch_axis) * chord_mm
        yc = yu * chord_mm * t_scale
        # поворот на угол установки: хорда наклоняется от плоскости вращения
        y = xc * cb - yc * sb      # окружное направление
        z = xc * sb + yc * cb      # осевое направление
        pts3d.append((r_mm, y, z))

    return cq.Wire.makePolygon([cq.Vector(*p) for p in pts3d], forConstruction=False)


# ----------------------------------------------------------------------
# Построение лопасти
# ----------------------------------------------------------------------
def build_blade(sections, hub_r, tip_round=True, pitch_axis=0.30):
    """Лофт по сечениям. sections — список dict с r_mm, chord_mm, beta_deg, thick_mm.

    Корневое сечение продлевается внутрь втулки, чтобы гарантировать
    объединение материала без зазора.
    """
    secs = sorted(sections, key=lambda s: s["r_mm"])

    # Продление корня внутрь втулки на 60% радиуса втулки —
    # обеспечивает надёжное булево объединение с бобышкой.
    root = dict(secs[0])
    root["r_mm"] = max(1.0, hub_r * 0.40)
    secs = [root] + secs

    if tip_round:
        # Схлопываем конец: узкое сечение сразу за последним для скругления
        tip = dict(secs[-1])
        tip["r_mm"] = secs[-1]["r_mm"] + max(0.4, secs[-1]["chord_mm"] * 0.05)
        tip["chord_mm"] = secs[-1]["chord_mm"] * 0.35
        tip["thick_mm"] = max(0.5, secs[-1]["thick_mm"] * 0.35)
        secs.append(tip)

    wires = [section_wire(s["r_mm"], s["chord_mm"], s["beta_deg"],
                          s["thick_mm"], pitch_axis) for s in secs]

    # ruled=False -> гладкая B-spline поверхность между сечениями
    solid = cq.Solid.makeLoft(wires, ruled=False)
    return cq.Workplane("XY").newObject([solid])


def build_hub(hub_d, axis_d, root_thick, blades, hub_h=None, chamfer=True):
    """Втулка (бобышка) с центральным отверстием под ось."""
    h = hub_h or max(root_thick * 1.9, hub_d * 0.34)
    hub = (cq.Workplane("XY")
           .circle(hub_d / 2.0)
           .extrude(h, both=True)
           .faces(">Z").fillet(min(1.5, h * 0.25)))
    # сквозное отверстие под вал
    hub = hub.faces(">Z").workplane(origin=(0, 0, 0)).hole(axis_d)
    return hub


def build_propeller(params, fillet_r=2.0, tip_round=True, pitch_axis=0.30):
    """Собирает винт: втулка + B лопастей, повёрнутых вокруг Z."""
    inp = params["input"]
    secs = params["sections"]
    B = int(inp["blades"])
    hub_d = float(inp["hub_dia_mm"])
    axis_d = float(inp["axis_dia_mm"])
    t_root = float(inp["root_thick_mm"])

    hub = build_hub(hub_d, axis_d, t_root, B)
    blade = build_blade(secs, hub_d / 2.0, tip_round, pitch_axis)

    result = hub
    for k in range(B):
        ang = 360.0 * k / B
        rot = blade.rotate((0, 0, 0), (0, 0, 1), ang)
        result = result.union(rot, clean=True)

    # Скругление в зоне перехода лопасть/втулка — снимает концентратор
    # напряжений, где печатный винт рвётся в первую очередь.
    if fillet_r and fillet_r > 0:
        try:
            zone = hub_d / 2.0 + max(secs[0]["chord_mm"], t_root) * 1.2
            edges = result.edges(
                cq.selectors.BoxSelector(
                    (-zone, -zone, -zone), (zone, zone, zone)))
            result = result.fillet(fillet_r)
            print(f"  скругление R{fillet_r} мм у корня — выполнено")
        except Exception as e:
            print(f"  ! скругление не применилось ({type(e).__name__}); "
                  f"сделайте его в CAD или уменьшите --fillet")

    return result


# ----------------------------------------------------------------------
# Демо-параметры (если нет JSON)
# ----------------------------------------------------------------------
def demo_params():
    D, hub_d, B, rpm, V0 = 280.0, 50.0, 3, 5000.0, 12.0
    R, rh = D / 2, hub_d / 2
    om = rpm * math.pi / 30
    c_ref, taper, aoa = 0.115 * R, 0.55, 5.0
    t_root, t_tip = 7.0, 1.6

    secs = []
    n = 24
    for i in range(n):
        xi = (i + 0.5) / n
        r = rh + (R - rh) * xi
        c = c_ref * (1 - taper * xi**2) * math.sqrt(max(0.25, 1 - xi**8 * 0.75))
        phi = math.atan2(V0 * 1.15, om * r / 1000.0)
        beta = math.degrees(phi) + aoa
        th = t_root + (t_tip - t_root) * xi**0.75
        secs.append({"r_mm": r, "chord_mm": c, "beta_deg": beta, "thick_mm": th})

    return {"meta": {"airfoil": "NACA 4412", "source": "demo"},
            "input": {"D_mm": D, "rpm": rpm, "blades": B, "V0_ms": V0,
                      "hub_dia_mm": hub_d, "axis_dia_mm": 6.0,
                      "root_thick_mm": t_root, "tip_thick_mm": t_tip},
            "sections": secs}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Генератор STEP-модели воздушного винта")
    ap.add_argument("json", nargs="?", help="propeller_params.json из калькулятора")
    ap.add_argument("-o", "--out", default="propeller.step", help="выходной STEP")
    ap.add_argument("--stl", metavar="FILE", help="дополнительно сохранить STL")
    ap.add_argument("--fillet", type=float, default=2.0,
                    help="радиус скругления у корня, мм (0 — отключить)")
    ap.add_argument("--pitch-axis", type=float, default=0.30,
                    help="положение оси поворота сечения по хорде (0..1)")
    ap.add_argument("--no-tip-round", action="store_true", help="не сужать конец")
    ap.add_argument("--tolerance", type=float, default=0.05, help="точность STL, мм")
    ap.add_argument("--demo", action="store_true", help="встроенные параметры")
    a = ap.parse_args()

    if a.demo or not a.json:
        print("Использую демо-параметры (D=280, 3 лопасти, 5000 об/мин)")
        params = demo_params()
    else:
        p = Path(a.json)
        if not p.exists():
            sys.exit(f"Файл не найден: {p}")
        params = json.loads(p.read_text(encoding="utf-8"))

    inp = params["input"]
    print(f"\nВинт: D={inp['D_mm']} мм, {inp['blades']} лопастей, "
          f"втулка {inp['hub_dia_mm']} мм, ось {inp['axis_dia_mm']} мм")
    print(f"Сечений: {len(params['sections'])}, профиль NACA 4412")
    print("Построение...")

    model = build_propeller(params, a.fillet,
                            not a.no_tip_round, a.pitch_axis)

    solid = model.val()
    vol = solid.Volume() / 1000.0
    bb = solid.BoundingBox()
    print(f"  объём: {vol:.2f} см³")
    print(f"  габарит: {bb.xlen:.1f} × {bb.ylen:.1f} × {bb.zlen:.1f} мм")

    # Оценка массы печати (плотности типовых нитей)
    for nm, rho in (("PETG", 1.27), ("PETG-CF", 1.30), ("PA-CF", 1.18)):
        print(f"  масса {nm}: {vol * rho:.1f} г (100% заполнение)")

    cq.exporters.export(model, a.out,
                        exportType="STEP",
                        opt={"write_pcurves": True, "precision_mode": 0})
    print(f"\nSTEP сохранён: {a.out}")

    if a.stl:
        cq.exporters.export(model, a.stl, exportType="STL",
                            tolerance=a.tolerance,
                            angularTolerance=0.1)
        print(f"STL сохранён:  {a.stl}  (допуск {a.tolerance} мм)")

    print("\nРекомендации по печати:")
    print("  - ориентация: лопасть на ребро или под 45°, НЕ плашмя")
    print("  - 100% заполнение либо 6+ периметров, слой 0.12-0.16 мм")
    print("  - отжиг PETG-CF при 120 C заметно поднимает межслойную прочность")
    print("  - обязательна балансировка перед раскруткой на полные обороты")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
create_qh_map.py — Quasi-harmonic mapping + интерполяция на регулярную сетку.

Аналог create_map.py, но вместо гармонического отображения использует
quasi-harmonic (invsqrt + phys + damping 1).

Pipeline для каждой .msh:
  read_msh() → harmonic_map (init) → QH_iter (invsqrt, phys, damp=1) →
  GridInterpolator (barycentric) → save x_data.csv, y_data.csv

Выход: ../train_qh/x_data.csv, ../train_qh/y_data.csv
  Каждая строка — flattened 2D массив (resolution×resolution), одна геометрия.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from read_msh import read_msh_file
from interpolate import GridInterpolator
from quasi_harmonic import QuasiHarmonicMapper

# Пути
BASE = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE, "../test_domains")
OUTPUT_DIR = os.path.join(BASE, "../train_qh")

RESOLUTION = 128
QH_MODE = 'invsqrt'
QH_EPERP = 'phys'
QH_DAMPING = 1
QH_NITER = 5


def process_mesh(msh_path, debug=True):
    """
    Полный pipeline для одной .msh:
      harmonic (init) → QH (invsqrt+phys) → interpolate → (Mx, My)
    """
    X, F, bnds = read_msh_file(msh_path)
    if X is None:
        return None, None
    if not bnds['bottom'] or not bnds['top']:
        print(f"  Skip: bad boundaries")
        return None, None

    # 1. Гармоническое отображение (инициализация)
    mapper_harm = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
    Y_harm = mapper_harm.build_mapping(verbose=False)

    # 2. Quasi-harmonic уточнение (insqrt + phys + damp)
    mapper_qh = QuasiHarmonicMapper(X, F, bnds,
                                    n_iter=QH_NITER,
                                    mode=QH_MODE,
                                    eperp_mode=QH_EPERP,
                                    damping=QH_DAMPING)
    Y_qh = mapper_qh.build_mapping(Y_init=Y_harm, verbose=debug)

    if np.any(np.isnan(Y_qh)):
        print("  NaN в QH, падаем на harmonic")
        mapper_harm.Y = Y_harm
        interpolator = GridInterpolator(mapper_harm)
    else:
        mapper_qh.Y = Y_qh
        mapper_qh.X = X
        mapper_qh.F = F
        interpolator = GridInterpolator(mapper_qh)

    # 3. Интерполяция на регулярную сетку
    M_x, M_y = interpolator.interpolate(sampling_size=RESOLUTION)

    # Лечение NaN на краях (ближайший сосед)
    if np.isnan(M_x).any():
        from scipy.interpolate import NearestNDInterpolator
        mask = ~np.isnan(M_x)
        coords = np.argwhere(mask)
        fill_x = NearestNDInterpolator(coords, M_x[mask])
        fill_y = NearestNDInterpolator(coords, M_y[mask])
        nan_coords = np.argwhere(np.isnan(M_x))
        for r, c in nan_coords:
            M_x[r, c] = fill_x(r, c)
            M_y[r, c] = fill_y(r, c)

    return M_x, M_y


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quasi-harmonic mapping pipeline")
    parser.add_argument("--input", "-i", default=INPUT_DIR,
                        help=f"Директория с .msh (default: {INPUT_DIR})")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR,
                        help=f"Директория для csv (default: {OUTPUT_DIR})")
    parser.add_argument("--resolution", "-r", type=int, default=RESOLUTION)
    parser.add_argument("--iter", "-n", type=int, default=QH_NITER)
    parser.add_argument("--damping", "-d", type=float, default=QH_DAMPING)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    resolution = args.resolution
    qh_niter = args.iter
    qh_damping = args.damping

    if not os.path.isdir(input_dir):
        print(f"Директория {input_dir} не найдена.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    out_x = os.path.join(output_dir, "x_data.csv")
    out_y = os.path.join(output_dir, "y_data.csv")

    # Чистим старые csv
    open(out_x, 'w').close()
    open(out_y, 'w').close()

    files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".msh") and "step" in f],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else x
    )

    print(f"Найдено {len(files)} .msh файлов в {input_dir}")
    print(f"QH: {QH_MODE} + {QH_EPERP}, damping={qh_damping}, "
          f"iter={qh_niter}, res={resolution}×{resolution}")

    for idx, fname in enumerate(files, 1):
        path = os.path.join(input_dir, fname)
        print(f"[{idx}/{len(files)}] {fname}...", end=" ", flush=True)
        try:
            gx, gy = process_mesh(path, debug=False)
            if gx is not None:
                with open(out_x, 'a') as fx, open(out_y, 'a') as fy:
                    np.savetxt(fx, gx.reshape(1, -1), delimiter=',')
                    np.savetxt(fy, gy.reshape(1, -1), delimiter=',')
                print("OK")
            else:
                print("FAIL")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nГотово. Обработано в {output_dir}/")
    print(f"  x_data: {out_x}")
    print(f"  y_data: {out_y}")


if __name__ == "__main__":
    main()
